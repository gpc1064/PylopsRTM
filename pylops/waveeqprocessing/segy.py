import segyio
import warnings
import numpy as np


__all__ = ['count_segy_shots', 'get_velocity_model', 'ReadSEGY2D', 'ReadSEGY3D']


def count_segy_shots(segy_path, shotattr=segyio.TraceField.FieldRecord):
    with segyio.open(segy_path, "r", ignore_geometry=True) as segyfile:
        headers = segyfile.header
        shotpoints = [trace[shotattr] for trace in headers]

        shot_ids = set(shotpoints)
        return len(shot_ids), list(shot_ids)


def get_velocity_model(model_path):
    """
    Read velocity model from a SEGY file
    """
    f = segyio.open(model_path, iline=segyio.tracefield.TraceField.FieldRecord,
                    xline=segyio.tracefield.TraceField.CDP)

    xl, il, t = f.xlines, f.ilines, f.samples
    if len(il) != 1:
        dims = (len(xl), len(il), len(t))
    else:
        dims = (len(xl), len(t))

    vp = f.trace.raw[:].reshape(dims)
    return vp, dims


class SegyDict(dict):
    
    DEFAULT_FIELDS = {
        "shot_id": segyio.TraceField.FieldRecord,
        "source_x": segyio.TraceField.SourceX,
        "source_y": segyio.TraceField.SourceY,
        "source_z": segyio.TraceField.SourceSurfaceElevation,
        "rec_x": segyio.TraceField.GroupX,
        "rec_y": segyio.TraceField.GroupY,
        "rec_z": segyio.TraceField.ReceiverGroupElevation,
        "scalco": segyio.TraceField.SourceGroupScalar
    }
    ALLOWED_KEYS = set(DEFAULT_FIELDS.keys())
    
    def __init__(self, input_dict=None):
        super().__init__(self.DEFAULT_FIELDS)
        
        if input_dict is not None:
            self._populate_from_dict(input_dict)
    
    def _populate_from_dict(self, input_dict):
        """
        Populates the instance from a user provided dict
        
        Args:
            input_dict (dict): source dict to populate
        """
        
        if not isinstance(input_dict, dict):
            raise TypeError(f"ReadSEGY: expecting a dict for segy_fields, but got {type(input_dict).__name__} instead")
        
        for key, value in input_dict.items():
            if key not in self.ALLOWED_KEYS:
                warnings.warn(
                    f"Key '{key}' will be ignored. "
                    f"Allowed keys: {sorted(self.ALLOWED_KEYS)}",
                    UserWarning,
                    stacklevel=2
                )
                continue
            
            if key not in self:
                self[key] = value
    
    def update(self, other=None, **kwargs):
        """
        Keep class restrictions.
        """
        if other is not None:
            self._populate_from_dict(other)
        if kwargs:
            self._populate_from_dict(kwargs)
    
    def __setitem__(self, key, value):
        """
        Keep class restrictions.
        """
        if key not in self.ALLOWED_KEYS:
            raise KeyError(
                f"Key '{key}' not allowed. "
                f"Allowed keys: {sorted(self.ALLOWED_KEYS)}"
            )
        super().__setitem__(key, value)
    
    def setdefault(self, key, default=None):
        """
        Keep class restrictions.
        """
        if key not in self.ALLOWED_KEYS:
            raise KeyError(
                f"Key '{key}' not allowed. "
                f"Allowed keys: {sorted(self.ALLOWED_KEYS)}"
            )
        return super().setdefault(key, default)


class ReadSEGY2D():

    def __init__(self, segy_path, mpi=None, shot_ids=None, segy_fields={}):

        self.segyfile = segy_path
        self.controller = mpi
        self.table, self.indexes = self.make_lookup_table(segy_path, mpi, shot_ids, segy_fields)
        self.isRecVariable = self._isRecVariable()
        self.nsrc = len(self.table)

    def _isRecVariable(self):
        """
        Verify if the number of receivers per shots is Regular.

        return True if all shots has the same number of receivers, and False otherwise
        """
        # get a set of values representing the number of traces per shot. In other words, the number os receivers per shot
        n_traces_per_shots = set(v["Num_Traces"] for v in self.table.values())

        # If the lenght of the set is 1, means that all the shots has the same number os receivers
        return len(n_traces_per_shots) != 1

    def make_lookup_table(self, sgy_file, mpi_controller, sampled_ids, segy_fields):
        '''
        Make a lookup of shots, where the keys are the shot record IDs being
        searched (looked up)

        Made by Oscar Mojica
        '''
        indexes = []
        lookup_table = {}

        samples = sampled_ids if not mpi_controller else mpi_controller.shot_ids
        target_fields = SegyDict(input_dict=segy_fields)
        
        with segyio.open(sgy_file, ignore_geometry=True) as f:
            index = None
            pos_in_file = 0

            for header in f.header:
                index = header[target_fields["shot_id"]]

                if (samples and (index not in samples)):
                    pos_in_file += 1
                    continue

                if int(header[target_fields["scalco"]]) < 0:
                    scalco = abs(1. / header[target_fields["scalco"]])
                else:
                    scalco = header[target_fields["scalco"]]
                # Esses comentários são temporários, scalel voltará a ser utilizado
                # if int(header[segyio.TraceField.ElevationScalar]) < 0:
                #     scalel = abs(1. / header[segyio.TraceField.ElevationScalar])
                # else:
                #     scalel = header[segyio.TraceField.ElevationScalar]
                # Check to see if we're in a new shot

                if index not in lookup_table.keys():
                    indexes.append(index)
                    lookup_table[index] = {}
                    lookup_table[index]['filename'] = sgy_file
                    lookup_table[index]['Trace_Position'] = pos_in_file
                    lookup_table[index]['Num_Traces'] = 1
                    lookup_table[index]['Source'] = (header[target_fields["source_x"]] * scalco,
                                                     header[target_fields["source_y"]] * scalco)
                    lookup_table[index]['Receivers'] = []
                    lookup_table[index]['scalcos'] = []
                else:  # Not in a new shot, so increase the number of traces in the shot by 1
                    lookup_table[index]['Num_Traces'] += 1
                lookup_table[index]['Receivers'].append((header[target_fields["rec_x"]] * scalco,
                                                         header[target_fields["rec_y"]] * scalco))
                lookup_table[index]['scalcos'].append(scalco)
                pos_in_file += 1

        return lookup_table, indexes

    def getsourceData(self, path):
        """
        Read source data from a SEGY file
        """
        f = segyio.open(path, iline=segyio.tracefield.TraceField.FieldRecord,
                        xline=segyio.tracefield.TraceField.CDP)

        src_data = f.trace.raw[:]
        return src_data

    def getVelocityModel(self, path):
        """
        Read velocity model from a SEGY file
        """
        return get_velocity_model(path)

    def getSourceCoords(self, index=0):
        src_coords = np.array(self.table[index]['Source'])
        sx = np.array([src_coords[0]])
        sz = np.array([src_coords[-1]])

        return sx, sz

    def getReceiverCoords(self, index=0):
        recs_coords = np.array(self.table[index]['Receivers'])
        rx = np.array([coord[0] for coord in recs_coords])
        rz = np.array([coord[-1] for coord in recs_coords])

        return rx, rz

    def getCoords(self, index=0):
        rec_coords = self.getReceiverCoords(index)
        src_coords = self.getSourceCoords(index)

        return src_coords, rec_coords

    def getTn(self):
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            num_samples = len(f.samples)
            samp_int = f.bin[segyio.BinField.Interval] / 1000

        return (num_samples - 1) * samp_int

    def getDt(self):
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            dt = f.bin[segyio.BinField.Interval]  # microseconds
        return dt / 1000  # return in miliseconds

    def getData(self, index: int):
        """
        Return the data from a specific index. It need to add a dimension to match returned data from _Wave
        Parameters
        ----------
        index : :obj:`int`
            Index of the shot that it will get the data
        """
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            position = self.table[index]['Trace_Position']
            traces_in_shot = self.table[index]['Num_Traces']

            num_samples = len(f.samples)
            retrieved_shot = np.zeros((1, traces_in_shot, num_samples), dtype=np.float32)

            shot_traces = f.trace[position:position + traces_in_shot]

            for ii, trace in enumerate(shot_traces):
                retrieved_shot[:, ii] = trace
        return retrieved_shot

    def getMinCoords(self):
        """
        Get the origin of the survey. It is the minimum value of the source and receiver coordinates
        """
        minX = np.inf
        minY = np.inf
        for isrc in self.indexes:
            src_coords, rec_coords = self.getCoords(isrc)

            minX = min(minX, np.min(src_coords[0]), np.min(rec_coords[0]))
            minY = min(minY, np.min(src_coords[1]), np.min(rec_coords[1]))

        return minX, minY


class ReadSEGY3D():

    def __init__(self, segy_path, mpi=None, shot_ids=None, segy_fields={}):

        self.segyfile = segy_path
        self.controller = mpi
        self.table, self.indexes = self.make_lookup_table(segy_path, mpi, shot_ids, segy_fields)
        self.isRecVariable = self._isRecVariable()
        self.nsrc = len(self.table)

    def make_lookup_table(self, sgy_file, mpi_controller, sampled_ids, segy_fields):
        '''
        Make a lookup of shots, where the keys are the shot record IDs being
        searched (looked up)

        Made by Oscar Mojica
        '''
        indexes = []
        lookup_table = {}

        samples = sampled_ids if not mpi_controller else mpi_controller.shot_ids
        target_fields = SegyDict(input_dict=segy_fields)

        with segyio.open(sgy_file, ignore_geometry=True) as f:
            index = None
            pos_in_file = 0

            for header in f.header:
                index = header[target_fields["shot_id"]]

                if (samples and (index not in samples)):
                    pos_in_file += 1
                    continue

                if int(header[target_fields["scalco"]]) < 0:
                    scalco = abs(1. / header[target_fields["scalco"]])
                else:
                    scalco = header[target_fields["scalco"]]
                # Esses comentários são temporários, scalel voltará a ser utilizado
                # if int(header[segyio.TraceField.ElevationScalar]) < 0:
                #     scalel = abs(1. / header[segyio.TraceField.ElevationScalar])
                # else:
                #     scalel = header[segyio.TraceField.ElevationScalar]
                # Check to see if we're in a new shot

                if index not in lookup_table.keys():
                    indexes.append(index)
                    lookup_table[index] = {}
                    lookup_table[index]['filename'] = sgy_file
                    lookup_table[index]['Trace_Position'] = pos_in_file
                    lookup_table[index]['Num_Traces'] = 1
                    lookup_table[index]['Source'] = (header[target_fields["source_x"]] * scalco,
                                                     header[target_fields["source_y"]] * scalco,
                                                     header[target_fields["source_z"]] * scalco)
                    lookup_table[index]['Receivers'] = []
                    lookup_table[index]['scalcos'] = []
                else:  # Not in a new shot, so increase the number of traces in the shot by 1
                    lookup_table[index]['Num_Traces'] += 1

                lookup_table[index]['Receivers'].append((header[target_fields["rec_x"]] * scalco,
                                                         header[target_fields["rec_y"]] * scalco,
                                                         header[target_fields["rec_z"]] * scalco))
                lookup_table[index]['scalcos'].append(scalco)
                pos_in_file += 1

        return lookup_table, indexes

    def _isRecVariable(self):
        """
        Verify if the number of receivers per shots is Regular.

        return True if all shots has the same number of receivers, and False otherwise
        """
        # get a set of values representing the number of traces per shot. In other words, the number os receivers per shot
        n_traces_per_shots = set(v["Num_Traces"] for v in self.table.values())

        # If the lenght of the set is 1, means that all the shots has the same number os receivers
        return len(n_traces_per_shots) != 1

    def getSourceCoords(self, index=0):
        src_coords = np.array(self.table[index]['Source'])
        sx = np.array([src_coords[0]])
        sy = np.array([src_coords[1]])
        sz = np.array([src_coords[-1]])

        return sx, sy, sz

    def getReceiverCoords(self, index=0):
        recs_coords = np.array(self.table[index]['Receivers'])
        rx = np.array([coord[0] for coord in recs_coords])
        ry = np.array([coord[1] for coord in recs_coords])
        rz = np.array([coord[-1] for coord in recs_coords])
        return rx, ry, rz

    def getCoords(self, index=0):
        rec_coords = self.getReceiverCoords(index)
        src_coords = self.getSourceCoords(index)

        return src_coords, rec_coords

    def getTn(self):
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            num_samples = len(f.samples)
            samp_int = f.bin[segyio.BinField.Interval] / 1000

        return (num_samples - 1) * samp_int

    def getDt(self):
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            dt = f.bin[segyio.BinField.Interval]  # microseconds
        return dt / 1000  # return in miliseconds

    def getData(self, index: int):
        """
        Return the data from a specific index. It need to add a dimension to match returned data from _Wave
        Parameters
        ----------
        index : :obj:`int`
            Index of the shot that it will get the data
        """
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            position = self.table[index]['Trace_Position']
            traces_in_shot = self.table[index]['Num_Traces']

            num_samples = len(f.samples)
            retrieved_shot = np.zeros((1, traces_in_shot, num_samples), dtype=np.float32)

            shot_traces = f.trace[position:position + traces_in_shot]

            for ii, trace in enumerate(shot_traces):
                retrieved_shot[:, ii] = trace
        return retrieved_shot

    def getMinCoords(self):
        """
        Get the origin of the survey. It is the minimum value of the source and receiver coordinates
        """
        minX = np.inf
        minY = np.inf
        minZ = np.inf
        for isrc in self.indexes:
            src_coords, rec_coords = self.getCoords(isrc)

            minX = min(minX, np.min(src_coords[0]), np.min(rec_coords[0]))
            minY = min(minY, np.min(src_coords[1]), np.min(rec_coords[1]))
            minZ = min(minZ, np.min(src_coords[-1]), np.min(rec_coords[-1]))

        return minX, minY, minZ
