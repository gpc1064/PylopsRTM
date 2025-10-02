import os
import numpy as np
from mpi4py import MPI
from typing import List, Optional, Tuple, Union, Any

__all__ = ["MPIShotsController"]


class MPIShotsController:
    """Controller for dividing and managing shots in an MPI environment.
    
    This class automates the distribution of seismic shots among MPI processes in
    seismic processing applications, ensuring load balancing and providing utilities
    for I/O operations and result combination.
    
    Attributes:
        shape (Tuple[int, int, int]):Model dimensions (nx, ny, nz)
        nsx (int):Number of shots in the x-direction
        nsy (int):Number of shots in the y-direction
        nbl (int): Thickness of the absorption boundary layer
        comm: MPI communicator
        size (int): Total number of MPI processes
        rank (int): Rank of the current process
        root (bool): True if the current process is rank 0
        global_shot_ids (Optional[List]): SEGY executions. Array with all shot IDs
        local_start (int): Starting index of local shots
        local_end (int): Ending index of local shots
        ns_local (int): Number of local shots
        shot_ids (Optional[List]): SEGY executions. IDs of shots assigned to the current process
    """
    
    def __init__(self, shape: Tuple[int, ...], nsx: int, nsy: int, nbl: int, 
                 comm: MPI.Comm, shot_ids: Optional[List] = None) -> None:
        """Initialize the MPI shots controller.
        
        Args:
            shape: Seismic model dimensions
            nsx: Number of shots in the x-direction
            nsy: Number of shots in the y-direction
            nbl: Thickness of the absorption boundary layer
            comm: MPI communicator
            shot_ids: Optional list of SEGY shot IDs
        """
        
        self.shape = shape
        self.nsx = nsx
        self.nsy = nsy
        self.nbl = nbl
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.root = self.rank == 0
        self.global_shot_ids = shot_ids

        ls, le, nsl = self.divide_sdomain()
        self.local_start = ls
        self.local_end = le
        self.ns_local = nsl
        if shot_ids:
            self.shot_ids = shot_ids[ls:le]
        else:
            self.shot_ids = None

    def _crop_stencil(self, m: List[np.ndarray], nbl: int) -> List[np.ndarray]:
        """
        Remove absorption boundary layers from models.
        
        Args:
            m: List of models/stencils to process
            nbl: Thickness of the boundary layer to remove
            
        Returns:
            List of cropped models without boundary layers
        """
        
        cropped_stencils = []
        for stencil in m:
            slices = tuple(slice(nbl, -nbl) for _ in range(stencil.ndim))
            cropped_stencils.append(stencil[slices])
        return cropped_stencils

    def divide_sdomain(self, M: Optional[int] = None, n: Optional[int] = None, 
                       idx: Optional[int] = None) -> Tuple[int, int, int]:
        """
        Divide the shot domain among MPI processes using a balanced approach.
        
        This method implements a round-robin distribution that ensures optimal
        load balancing by distributing any remainder shots to the first processes.
        
        Args:
            M: Total number of points to divide (default: self.nsx * self.nsy)
            n: Number of processes (default: self.size)
            idx: Process rank (default: self.rank)
            
        Returns:
            Tuple containing:
                - start: Starting index for this process
                - end: Ending index for this process
                - ns_local: Number of shots assigned to this process
        """
        points = M or (self.nsx * self.nsy)
        size = n or self.size
        rank = idx or self.rank

        base = points // size
        remainder = points % size

        if rank < remainder:
            start = rank * (base + 1)
            end = start + (base + 1)
        else:
            start = remainder * (base + 1) + (rank - remainder) * base
            end = start + base

        return start, end, end - start

    def divide_shots(self, spacing: Tuple[float, ...], dpf: float = .05, 
                     mode: str = "centered_uniform") -> np.ndarray:
        """
        Calculate shot coordinates for local shots based on distribution mode.
        
        Args:
            spacing: Grid spacing in each dimension (dx, dy, dz)
            dpf: Depth percentage factor (default: 0.05 = 5% of total depth)
            mode: Coordinates distribution mode 
            
        Returns:
            Array of shot coordinates [x, y, z] for local shots
        """
        
        nsx = self.nsx
        nsy = self.nsy
        nx, ny, nz = tuple(sh * sp for sh, sp in zip(self.shape, spacing))
        depth = int(dpf * nz)

        local_start, local_end, ns_local = self.divide_sdomain()

        local_sources = []

        if mode == "uniform":
            dx = (nx - 1) / (max(nsx - 1, 1)) if nsx > 1 else nx / 2
            dy = (ny - 1) / (max(nsy - 1, 1)) if nsy > 1 else ny / 2

            for sindex in range(local_start, local_end):
                iy = sindex // nsx
                ix = sindex % nsx

                x = ix * dx if nsx > 1 else dx
                y = iy * dy if nsy > 1 else dy
                z = depth

                local_sources.append([x, y, z])
        elif mode == "centered_uniform":
            d_sx = nx / (nsx + 1)
            d_sy = ny / (nsy + 1)
            for sindex in range(local_start, local_end):
                iy = sindex // nsx
                ix = sindex % nsx

                x = (ix + 1) * d_sx
                y = (iy + 1) * d_sy
                z = depth

                local_sources.append([x, y, z])

        self.local_start = local_start
        self.local_end = local_end
        self.ns_local = ns_local
        return np.array(local_sources)

    def read_data(self, pwd: str, seismic_nt: int, mode: str = "shot_number",
                  recs_names: List[str] = ["recX", "recY", "recZ"]) -> List[List[np.ndarray]]:
        """
        Read seismic data from binary files for local shots.
        
        Args:
            pwd: Path to the directory containing data files
            seismic_nt: Number of time samples in seismic data
            mode: Reading mode (currently only "shot_number" supported)
            recs_names: List of receiver component names to read
            
        Returns:
            Nested list of seismic data arrays organized by [component][shot]
            
        Note:
            Files are expected to be named as {rec_name}_{sx}_{sy}.bin
        """
        
        lstart = self.local_start
        lend = self.local_end
        nsx = self.nsx
        nx, ny, nz = self.shape

        dims_recs = []
        for _ in recs_names:
            dims_recs.append([])

        for sindex in range(lstart, lend):
            sy = sindex // nsx
            sx = sindex % nsx
            for ri, rname in enumerate(recs_names):
                filename = f"{rname}_{sx}_{sy}.bin"
                aux = open(os.path.join(pwd, filename), "rb")
                data = np.fromfile(aux, dtype=np.float32, count=nx * ny * seismic_nt)
                aux.close()
                data = data.reshape([nx * ny, seismic_nt])
                dims_recs[ri].append(data)

        return dims_recs

    def write_data(self, recs_data: List[List[np.ndarray]], pwd: str, 
                   mode: str = "shot_number",
                   recs_names: List[str] = ["recX", "recY", "recZ"]) -> None:
        """
        Write seismic data to binary files for local shots.
        
        Args:
            recs_data: Nested list of seismic data arrays organized by [component][shot]
            pwd: Path to the directory for output files
            mode: Writing mode (currently only "shot_number" supported)
            recs_names: List of receiver component names for file naming
            
        Note:
            Files will be named as {rec_name}_{sx}_{sy}.bin
        """
        lstart = self.local_start
        lend = self.local_end
        nsx = self.nsx

        shot = 0
        for sindex in range(lstart, lend):
            sy = sindex // nsx
            sx = sindex % nsx
            for ri, rname in enumerate(recs_names):
                filename = f"{rname}_{sx}_{sy}.bin"
                aux = open(os.path.join(pwd, filename), "wb")
                rec = recs_data[ri][shot]
                rec.tofile(aux)
                aux.close()
            shot += 1

    def build_result(self, local_image: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Combine local results from all processes using MPI Allreduce.
        
        Args:
            local_image: Local image data to combine
            
        Returns:
            Global combined image from all processes
        """
        shape = self.shape
        comm = self.comm

        if not isinstance(local_image, np.ndarray):
            local_image = np.array(local_image)

        result_sum = []
        for i in range(len(local_image)):
            result_dim = np.zeros(shape, dtype=np.float32)
            comm.Allreduce(local_image[i], result_dim, op=MPI.SUM)
            result_sum.append(result_dim)

        return np.array(result_sum)
