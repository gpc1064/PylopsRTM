from copy import deepcopy
from typing import Callable, TypeVar, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from pylops.utils.twowaympi import MPIShotsController
from pylops.waveeqprocessing.twoway import _Wave
from pylops.waveeqprocessing.segy import ReadSEGY2D, ReadSEGY3D, count_segy_shots


devito_message = deps.devito_import("the twoway module")

if devito_message is None:
    from devito import Function
    from devito.builtins import initialize_function

    from examples.seismic import AcquisitionGeometry, Model
    from examples.seismic.acoustic import AcousticWaveSolver
    from examples.seismic.multiparameter.viscoacoustic import (
        ViscoacousticWaveSolver as ViscoacousticWaveSolverMulti,
    )
    from examples.seismic.viscoacoustic import ViscoacousticWaveSolver

MPIComm = TypeVar("mpi4py.MPI.Comm")


class _ViscoAcousticWave(_Wave):
    """Devito ViscoAcoustic propagator.

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
    qp : :obj:`numpy.ndarray
        P-wave attenuation
    b : :obj:`numpy.ndarray
        Buoyancy
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
        Initial time
    tn : :obj:`int`
        Number of time samples
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    kernel :
        selects a visco-acoustic equation from the options below:
                'sls' (Standard Linear Solid) :
                1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
                viscoacoustic equation
                2nd order - Bai et al. (2014) viscoacoustic equation
                'kv' - Ren et al. (2014) viscoacoustic equation
                'maxwell' - Deng and McMechan (2007) viscoacoustic equation
                Defaults to 'sls' 2nd order.
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

    _solver_type = ViscoacousticWaveSolver

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        qp: NDArray,
        b: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        kernel: str = "sls",
        time_order: int = 2,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "fwd",
        src_y: NDArray = None,
        rec_y: NDArray = None,
        dt: int = None,
        segy_path: str = None,
        segy_mpi: MPIComm = None,
        segy_sample: Union[int, float] = None,
        segy_fields: dict = {},
        mpi_instant_reduce: bool = False,
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
        self._create_model(shape, origin, spacing, vp, qp, b, space_order, nbl, dt)
        self._create_geometry(
            src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0
        )
        self.checkpointing = checkpointing
        self.kernel = kernel
        self.time_order = time_order
        self.karguments = {}
        self.op_name = op_name

        if (segy_path):
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
                controller = MPIShotsController(shape, sample, nsy, nbl, segy_mpi, shot_ids=sampled_sids)
                self.mpi_controller = controller

            self.segyReader = SegyReader(
                segy_path,
                mpi=getattr(self, "mpi_controller", None),
                shot_ids=sampled_sids,
                segy_fields=segy_fields
            )

        self.instant_reduce = mpi_instant_reduce

        dims = self._compute_dims(vp.shape)

        super().__init__(
            dtype=np.dtype(dtype),
            dims=dims,
            dimsd=(len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )
        self._register_multiplications(op_name)

    def _compute_dims(self, grid_shape):
        return grid_shape

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        qp: NDArray,
        b: NDArray,
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
        qp : :obj:`numpy.ndarray
            P-wave attenuation
        b : :obj:`numpy.ndarray
            Buoyancy
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries

        """
        self.space_order = space_order
        self.model = Model(
            space_order=space_order,
            vp=vp * 1e-3,
            qp=qp,
            b=b,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            dt=dt,
        )

    def _born_oneshot(self, solver: ViscoacousticWaveSolver, dm: NDArray) -> NDArray:
        """Born modelling for one shot

        Parameters
        ----------
        solver : :obj:`ViscoacousticWaveSolver`
            Devito's solver object.
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        dmext = np.zeros(self.model.grid.shape, dtype=np.float32)

        nbl = self.model.nbl
        slices = tuple(slice(nbl, -nbl) for _ in range(dmext.ndim))
        dmext[slices] = dm

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]

        d = solver.jacobian(dmext, src=None if not hasattr(self, "wav") else self.wav)[
            0
        ]
        # d = solver.jacobian(src=None if not hasattr(self, "wav") else self.wav, **self.karguments)[0]
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

        # create solver
        solver = self._solver_type(
            self.model,
            geometry,
            space_order=self.space_order,
            kernel=self.kernel,
            time_order=self.time_order,
        )

        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._born_oneshot(solver, dm)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

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
        solver = self._solver_type(
            self.model,
            geometry,
            space_order=self.space_order,
            kernel=self.kernel,
            time_order=self.time_order,
        )
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._fwd_oneshot(solver, v)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _grad_oneshot(self, solver: ViscoacousticWaveSolver, isrc, dobs):

        rec = self.geometry.rec.copy()
        rec.data[:] = dobs.T

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]

        # source wavefield
        if hasattr(self, "src_wavefield"):
            p = self.src_wavefield[isrc]
        else:
            p = solver.forward(save=True, src=None if not hasattr(self, "wav") else self.wav)[1]

        # adjoint modelling (reverse wavefield plus imaging condition)
        grad, _ = solver.jacobian_adjoint(rec, p)

        return grad

    def _grad_allshots(self, dobs: NDArray) -> NDArray:
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

        solver = ViscoacousticWaveSolver(
            self.model,
            geometry,
            space_order=self.space_order,
            kernel=self.kernel,
            time_order=self.time_order,
        )

        nsrc = self.geometry.src_positions.shape[0]
        mtot = np.zeros(self.model.shape, dtype=np.float32)

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            grad = self._grad_oneshot(solver, isrc, dobs[isrc])
            mtot += self._crop_model(grad.data, self.model.nbl)

        controller = getattr(self, "mpi_controller", None)
        if (controller and self.instant_reduce):
            return controller.build_result([mtot])[0]
        else:
            return mtot

    def _register_multiplications(self, op_name: str) -> None:
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
        if op_name == "born":
            self._acoustic_matvec = self._born_allshots
        self._acoustic_rmatvec = self._grad_allshots

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_rmatvec(x)
        return y


class _ViscoMultiparameterWave(_ViscoAcousticWave):
    _solver_type = ViscoacousticWaveSolverMulti

    def _compute_dims(self, grid_shape):
        # Determine the number of outputs based on the modeling type
        _dims_table = {"fwd": 1, "born": 2}
        if self.op_name not in _dims_table:
            raise TypeError("Provided op_name '%s' is not valid" % self.op_name)

        if _dims_table[self.op_name] == 1:
            return grid_shape
        else:
            return (_dims_table[self.op_name], *grid_shape)

    def _born_oneshot(
        self, solver: ViscoacousticWaveSolverMulti, data: NDArray
    ) -> NDArray:
        """Born modelling for one shot

        Parameters
        ----------
        solver : :obj:`ViscoacousticWaveSolverMulti`
            Devito's solver object.
        data : :obj:`np.ndarray`
            Contain dm and dtau

            data[0] = dm
            data[1] = dtau

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        dmext = np.zeros(self.model.grid.shape, dtype=np.float32)
        dtauext = np.zeros(self.model.grid.shape, dtype=np.float32)

        nbl = self.model.nbl
        slices = tuple(slice(nbl, -nbl) for _ in range(dmext.ndim))
        dmext[slices] = data[0]
        dtauext[slices] = data[1]

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]

        d = solver.jacobian(
            dmext, dtauext, src=None if not hasattr(self, "wav") else self.wav
        )[0]
        # d = solver.jacobian(src=None if not hasattr(self, "wav") else self.wav, **self.karguments)[0]
        d = d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
        return d

    def _grad_oneshot(self, solver: ViscoacousticWaveSolverMulti, isrc, rec_data):

        rec = self.geometry.rec.copy()
        rec.data[:] = rec_data.T[:]

        # assign source location to source object with custom wavelet
        if hasattr(self, "wav"):
            self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]

        # source wavefield
        if hasattr(self, "src_wavefield"):
            p = self.src_wavefield[isrc]
        else:
            p = solver.forward(save=True)[1]

        # adjoint modelling (reverse wavefield plus imaging condition)
        grad_m, grad_tau, _ = solver.jacobian_adjoint(rec, p)

        return grad_m, grad_tau

    def _grad_allshots(self, rec_data: NDArray) -> NDArray:
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

        solver = ViscoacousticWaveSolverMulti(
            self.model,
            geometry,
            space_order=self.space_order,
            kernel=self.kernel,
            time_order=self.time_order,
        )

        nsrc = self.geometry.src_positions.shape[0]
        mtot = np.zeros((2, *self.model.shape), dtype=np.float32)

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            grad_m, grad_tau = self._grad_oneshot(solver, isrc, rec_data[isrc])
            mtot[0] += self._crop_model(grad_m.data, self.model.nbl)
            mtot[1] += self._crop_model(grad_tau.data, self.model.nbl)
        return mtot

    def _register_multiplications(self, op_name: str) -> None:
        if op_name == "born":
            self._viscoMulti_matvec = self._born_allshots
        if op_name == "fwd":
            self._viscoMulti_matvec = self._fwd_allshots
        self._viscoMulti_rmatvec = self._grad_allshots

    def adjoint(self):
        """
        If the direct modeling operation is forward, it is necessary to adapt the dimensions to make them
        compatible with the gradient output.
        """
        # Check if the operator's name is 'fwd' to determine if forward modeling is being performed
        if self.op_name == "fwd":
            Op = deepcopy(self)
            new_dims = (2, *Op.dims)
            # Updates input dimensions to reflect the extra channel expected by the gradient output
            Op._update_dimensions(new_dims, Op.dimsd)
            return LinearOperator.adjoint(Op)
        return LinearOperator.adjoint(self)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._viscoMulti_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._viscoMulti_rmatvec(x)
        return y

    H: Callable[[LinearOperator], LinearOperator] = property(adjoint)


ViscoAcousticWave2D = _ViscoAcousticWave
ViscoAcousticWave3D = _ViscoAcousticWave
MultiparameterViscoAcoustic2D = _ViscoMultiparameterWave
MultiparameterViscoAcoustic3D = _ViscoMultiparameterWave
