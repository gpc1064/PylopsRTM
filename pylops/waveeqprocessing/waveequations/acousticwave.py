from typing import Tuple, TypeVar, Union

import numpy as np

from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.twowaympi import MPIShotsController
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from pylops.waveeqprocessing.segy import ReadSEGY2D, ReadSEGY3D, count_segy_shots
from pylops.waveeqprocessing.twoway import _Wave

devito_message = deps.devito_import("the twoway module")

if devito_message is None:
    from devito import Function
    from devito.builtins import initialize_function

    from examples.seismic import AcquisitionGeometry, Model
    from examples.seismic.acoustic import AcousticWaveSolver

MPIComm = TypeVar("mpi4py.MPI.Comm")


class _AcousticWave(_Wave):
    """Devito Acoustic propagator.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
        Model spacing ``(dx, dz)``
    vp : :obj:`numpy.ndarray`
        Velocity model in m/s
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_y : :obj:`numpy.ndarray`
        Source y-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_y : :obj:`numpy.ndarray`
        Receiver y-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time in ms
    tn : :obj:`int`
        Final time in ms
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
        but it will slow down computations
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "born",
        src_y: NDArray = None,
        rec_y: NDArray = None,
        dt: int = None,
        segy_path: str = None,
        segy_mpi: MPIComm = None,
        segy_sample: Union[int, float] = None,
        segy_fields: dict = {},
        mpi_instant_reduce: bool = False,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_path: str = None,
        dswap_compression: str = None,
        dswap_compression_value: float | int = None,
        dswap_verbose: bool = False,
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        is_2d = len(shape) == 2
        is_3d = len(shape) == 3

        if is_2d and (rec_y is not None or src_y is not None):
            raise Exception(
                "Attempting to create a 3D operator using a 2D intended class!"
            )

        if is_3d and (rec_y is None or src_y is None):
            raise Exception(
                "Attempting to create a 2D operator using a 3D intended class!"
            )

        # create model
        self._create_model(shape, origin, spacing, vp, space_order, nbl, dt)
        self._create_geometry(
            src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0
        )
        self.checkpointing = checkpointing
        self.karguments = {}

        if segy_path:
            SegyReader = ReadSEGY3D if is_3d else ReadSEGY2D

            nshots, shot_ids = count_segy_shots(segy_path)
            nsy = 1

            sample = segy_sample or nshots
            if sample <= 0 or sample > nshots:
                raise Exception("segy sample must be between (0," + str(nshots) + "]")
            elif sample >= 1:
                # Straight number of samples
                sample = int(sample)
            else:
                # Percentage
                sample = int(nshots * sample)

            idxs = np.linspace(0, nshots - 1, num=sample, dtype=int)
            sampled_sids = [shot_ids[i] for i in idxs]

            if segy_mpi:
                controller = MPIShotsController(
                    shape, sample, nsy, nbl, segy_mpi, shot_ids=sampled_sids
                )
                self.mpi_controller = controller

            self.segyReader = SegyReader(
                segy_path,
                mpi=getattr(self, "mpi_controller", None),
                shot_ids=sampled_sids,
                segy_fields=segy_fields
            )

        self.instant_reduce = mpi_instant_reduce

        self._dswap_opt = {
            "dswap": dswap,
            "dswap_disks": dswap_disks,
            "dswap_folder": dswap_folder,
            "dswap_path": dswap_path,
            "dswap_compression": dswap_compression,
            "dswap_compression_value": dswap_compression_value,
            "dswap_verbose": dswap_verbose,
        }

        super().__init__(
            dtype=np.dtype(dtype),
            dims=vp.shape,
            dimsd=(len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )
        self._register_multiplications(op_name)

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 6,
        nbl: int = 20,
        dt: int = None,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in m/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries

        """
        self.space_order = space_order
        self.model = Model(
            space_order=space_order,
            vp=vp * 1e-3,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
            dt=dt,
        )

    def _srcillumination_oneshot(self, isrc: int) -> Tuple[NDArray, NDArray]:
        """Source wavefield and illumination for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model

        Returns
        -------
        u0 : :obj:`np.ndarray`
            Source wavefield
        src_ill : :obj:`np.ndarray`
            Source illumination

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[isrc, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )
        solver = AcousticWaveSolver(
            self.model, geometry, space_order=self.space_order, **self._dswap_opt
        )

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]

        # source wavefield
        u0 = solver.forward(
            save=True, src=None if not hasattr(self, "wav") else self.wav
        )[1]

        # source illumination
        src_ill = self._crop_model((u0.data**2).sum(axis=0), self.model.nbl)
        return u0, src_ill

    def srcillumination_allshots(self, savewav: bool = False) -> None:
        """Source wavefield and illumination for all shots

        Parameters
        ----------
        savewav : :obj:`bool`, optional
            Save source wavefield (``True``) or not (``False``)

        """
        nsrc = self.geometry.src_positions.shape[0]
        if savewav:
            self.src_wavefield = []
        self.src_illumination = np.zeros(self.model.shape)

        for isrc in range(nsrc):
            src_wav, src_ill = self._srcillumination_oneshot(isrc)
            if savewav:
                self.src_wavefield.append(src_wav)
            self.src_illumination += src_ill

    def _born_oneshot(self, solver: AcousticWaveSolver, dm: NDArray) -> NDArray:
        """Born modelling for one shot

        Parameters
        ----------
        solver : :obj:`AcousticWaveSolver`
            Devito's solver object.
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """

        # set perturbation
        dmext = np.zeros(self.model.grid.shape, dtype=np.float32)
        if dmext.ndim == 2:
            dmext[
                self.model.nbl : -self.model.nbl,
                self.model.nbl : -self.model.nbl,
            ] = dm
        else:
            dmext[
                self.model.nbl : -self.model.nbl,
                self.model.nbl : -self.model.nbl,
                self.model.nbl : -self.model.nbl,
            ] = dm

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]

        d = solver.jacobian(dmext, src=None if not hasattr(self, "wav") else self.wav)[
            0
        ]
        d = d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
        return d

    def _born_allshots(self, dm: NDArray) -> NDArray:
        """Born modelling for all shots

        Parameters
        -----------
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        # solve
        solver = AcousticWaveSolver(
            self.model, geometry, space_order=self.space_order, **self._dswap_opt
        )

        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._born_oneshot(solver, dm)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _bornadj_oneshot(self, solver: AcousticWaveSolver, isrc, dobs):
        """Adjoint born modelling for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        dobs : :obj:`np.ndarray`
            Observed data to inject

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        # set disk_swap bool
        dswap = self._dswap_opt.get("dswap", False)

        # create boundary data
        recs = self.geometry.rec.copy()
        recs.data[:] = dobs.T[:]

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]

        # source wavefield
        if hasattr(self, "src_wavefield"):
            u0 = self.src_wavefield[isrc]
        else:
            u0 = solver.forward(
                save=True if not dswap else False,
                src=None if not hasattr(self, "wav") else self.wav,
            )[1]

        # adjoint modelling (reverse wavefield plus imaging condition)
        model = solver.jacobian_adjoint(
            rec=recs, u=u0, checkpointing=self.checkpointing
        )[0]
        return model

    def _bornadj_allshots(self, dobs: NDArray) -> NDArray:
        """Adjoint Born modelling for all shots

        Parameters
        ----------
        dobs : :obj:`np.ndarray`
            Observed data to inject

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        solver = AcousticWaveSolver(
            self.model, geometry, space_order=self.space_order, **self._dswap_opt
        )

        nsrc = self.geometry.src_positions.shape[0]
        mtot = np.zeros(self.model.shape, dtype=np.float32)

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            m = self._bornadj_oneshot(solver, isrc, dobs[isrc])
            mtot += self._crop_model(m.data, self.model.nbl)

        controller = getattr(self, "mpi_controller", None)
        if controller and self.instant_reduce:
            return controller.build_result([mtot])[0]
        else:
            return mtot

    def _fwd_oneshot(self, solver: AcousticWaveSolver, v: NDArray) -> NDArray:
        """Forward modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        # create function representing the physical parameter received as parameter
        function = Function(
            name="vp",
            grid=self.model.grid,
            space_order=self.model.space_order,
            parameter=True,
        )

        # Assignment of values to physical parameters functions based on the values in 'v'
        initialize_function(function, v, self.model.padsizes)

        # add vp to karguments to be used inside devito's solver
        self.karguments.update({"vp": function})

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]

        d = solver.forward(**self.karguments, src=None if not hasattr(self, "wav") else self.wav)[0]
        d = d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
        return d

    def _fwd_allshots(self, v: NDArray) -> NDArray:
        """Forward modelling for all shots

        Parameters
        -----------
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        # solve
        solver = AcousticWaveSolver(
            self.model,
            geometry,
            space_order=self.space_order,
        )

        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._fwd_oneshot(solver, v)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _register_multiplications(self, op_name: str) -> None:
        if op_name == "born":
            self._acoustic_matvec = self._born_allshots
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
        self._acoustic_rmatvec = self._bornadj_allshots

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_rmatvec(x)
        return y


AcousticWave2D = _AcousticWave
AcousticWave3D = _AcousticWave
