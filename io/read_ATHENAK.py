"""
Reader for AthenaK binary outputs with stitching to (x, y, z) order.
"""

import os
import struct
import numpy as np
try:
    from numba import njit, prange, types
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def read_binary(filename: str) -> dict:
    """
    Minimal reader for AthenaK .bin outputs.
    """
    filedata = {}
    with open(filename, "rb") as fp:
        fp.seek(0, 2)
        filesize = fp.tell()
        fp.seek(0, 0)

        code_header = fp.readline().split()
        if len(code_header) < 1 or code_header[0] != b"Athena":
            raise TypeError("unknown file format")
        version = code_header[-1].split(b"=")[-1]
        if version != b"1.1":
            raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

        pheader_count = int(fp.readline().split(b"=")[-1])
        pheader = {}
        for _ in range(pheader_count - 1):
            key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=")]
            pheader[key] = val
        time = float(pheader["time"])
        cycle = int(pheader["cycle"])
        locsizebytes = int(pheader["size of location"])
        varsizebytes = int(pheader["size of variable"])

        nvars = int(fp.readline().split(b"=")[-1])
        var_list = [v.decode("utf-8") for v in fp.readline().split()[1:]]
        header_size = int(fp.readline().split(b"=")[-1])
        header = [
            line.decode("utf-8").split("#")[0].strip()
            for line in fp.read(header_size).split(b"\n")
        ]
        header = [line for line in header if len(line) > 0]

        if locsizebytes not in [4, 8]:
            raise ValueError(f"unsupported location size (in bytes) {locsizebytes}")
        if varsizebytes not in [4, 8]:
            raise ValueError(f"unsupported variable size (in bytes) {varsizebytes}")

        locfmt = "d" if locsizebytes == 8 else "f"
        varfmt = "d" if varsizebytes == 8 else "f"

        def get_from_header(header, blockname, keyname):
            blockname = blockname.strip()
            keyname = keyname.strip()
            if not blockname.startswith("<"):
                blockname = "<" + blockname
            if blockname[-1] != ">":
                blockname += ">"
            block = "<none>"
            for line in [entry for entry in header]:
                if line.startswith("<"):
                    block = line
                    continue
                key, value = line.split("=")
                if block == blockname and key.strip() == keyname:
                    return value
            raise KeyError(f"no parameter called {blockname}/{keyname}")

        Nx1 = int(get_from_header(header, "<mesh>", "nx1"))
        Nx2 = int(get_from_header(header, "<mesh>", "nx2"))
        Nx3 = int(get_from_header(header, "<mesh>", "nx3"))
        nx1 = int(get_from_header(header, "<meshblock>", "nx1"))
        nx2 = int(get_from_header(header, "<meshblock>", "nx2"))
        nx3 = int(get_from_header(header, "<meshblock>", "nx3"))

        nghost = int(get_from_header(header, "<mesh>", "nghost"))

        x1min = float(get_from_header(header, "<mesh>", "x1min"))
        x1max = float(get_from_header(header, "<mesh>", "x1max"))
        x2min = float(get_from_header(header, "<mesh>", "x2min"))
        x2max = float(get_from_header(header, "<mesh>", "x2max"))
        x3min = float(get_from_header(header, "<mesh>", "x3min"))
        x3max = float(get_from_header(header, "<mesh>", "x3max"))

        n_vars = len(var_list)
        mb_count = 0

        mb_index = []
        mb_logical = []
        mb_geometry = []

        mb_data = {var: [] for var in var_list}

        while fp.tell() < filesize:
            mb_index.append(np.array(struct.unpack("@6i", fp.read(24))) - nghost)
            nx1_out = (mb_index[mb_count][1] - mb_index[mb_count][0]) + 1
            nx2_out = (mb_index[mb_count][3] - mb_index[mb_count][2]) + 1
            nx3_out = (mb_index[mb_count][5] - mb_index[mb_count][4]) + 1

            mb_logical.append(np.array(struct.unpack("@4i", fp.read(16))))
            mb_geometry.append(
                np.array(struct.unpack("=6" + locfmt, fp.read(6 * locsizebytes)))
            )

            data = np.array(
                struct.unpack(
                    f"={nx1_out*nx2_out*nx3_out*n_vars}" + varfmt,
                    fp.read(varsizebytes * nx1_out * nx2_out * nx3_out * n_vars),
                )
            )
            data = data.reshape(n_vars, nx3_out, nx2_out, nx1_out)
            for vari, var in enumerate(var_list):
                mb_data[var].append(data[vari])
            mb_count += 1

    filedata["header"] = header
    filedata["time"] = time
    filedata["cycle"] = cycle
    filedata["var_names"] = var_list

    filedata["Nx1"] = Nx1
    filedata["Nx2"] = Nx2
    filedata["Nx3"] = Nx3
    filedata["nvars"] = nvars

    filedata["x1min"] = x1min
    filedata["x1max"] = x1max
    filedata["x2min"] = x2min
    filedata["x2max"] = x2max
    filedata["x3min"] = x3min
    filedata["x3max"] = x3max

    filedata["n_mbs"] = mb_count
    filedata["nx1_mb"] = nx1
    filedata["nx2_mb"] = nx2
    filedata["nx3_mb"] = nx3
    filedata["nx1_out_mb"] = (mb_index[0][1] - mb_index[0][0]) + 1
    filedata["nx2_out_mb"] = (mb_index[0][3] - mb_index[0][2]) + 1
    filedata["nx3_out_mb"] = (mb_index[0][5] - mb_index[0][4]) + 1

    filedata["mb_index"] = np.array(mb_index)
    filedata["mb_logical"] = np.array(mb_logical)
    filedata["mb_geometry"] = np.array(mb_geometry)
    filedata["mb_data"] = mb_data

    return filedata


class AthenaKBinData:
    """
    Container for stitched AthenaK bin outputs with lazy loading.
    """

    def __init__(self, file_number: int, basename: str = "CS", data_dir: str = "."):
        self.file_number = file_number
        self.basename = basename
        self.data_dir = data_dir
        self.w_binary_fname = self._default_fname("mhd_w")
        self.b_binary_fname = self._default_fname("mhd_bcc")
        self.cur_binary_fname = self._default_fname("mhd_jz")

        self.meta_w = None
        self.meta_b = None
        self.dens = None
        self.vel = None
        self.press = None
        self.mag = None
        self.cur = None
        self._loaded_w = set()
        self._cache_w = {}

    def _default_fname(self, kind: str) -> str:
        return os.path.join(
            self.data_dir,
            f"{self.basename}.{kind}.{self.file_number:05d}.bin",
        )

    @staticmethod
    def _get_param_from_header(filedata, block_name: str, key: str, cast=float):
        block = None
        for line in filedata["header"]:
            if line.startswith("<"):
                block = line
                continue
            if block == f"<{block_name}>":
                if line.strip().startswith(f"{key}"):
                    _, val = line.split("=")
                    return cast(val.strip())
        raise KeyError(f"{block_name}/{key} not found in header")

    def _convert_and_stack(self, binary_fname: str, variables=None) -> dict:
        filedata = read_binary(binary_fname)

        nx1_mb = filedata["nx1_out_mb"]
        nx2_mb = filedata["nx2_out_mb"]
        nx3_mb = filedata["nx3_out_mb"]

        lx1 = filedata["mb_logical"][:, 0]
        lx2 = filedata["mb_logical"][:, 1]
        lx3 = filedata["mb_logical"][:, 2]

        lx1_min, lx1_max = lx1.min(), lx1.max()
        lx2_min, lx2_max = lx2.min(), lx2.max()
        lx3_min, lx3_max = lx3.min(), lx3.max()

        nx1_global = (lx1_max - lx1_min + 1) * nx1_mb
        nx2_global = (lx2_max - lx2_min + 1) * nx2_mb
        nx3_global = (lx3_max - lx3_min + 1) * nx3_mb

        vars_to_use = variables if variables is not None else filedata["var_names"]
        missing = [v for v in vars_to_use if v not in filedata["var_names"]]
        if missing:
            raise ValueError(f"Requested variables not in file: {missing}")

        stacked = {}
        for name in vars_to_use:
            dtype = filedata["mb_data"][name][0].dtype
            stacked[name] = np.zeros((nx1_global, nx2_global, nx3_global), dtype=dtype)

        if _NUMBA_AVAILABLE:
            # numba-accelerated stitching
            for name in vars_to_use:
                blocks = np.array(filedata["mb_data"][name])
                self._stitch_blocks_numba(
                    stacked[name],
                    blocks,
                    filedata["mb_logical"],
                    lx1_min,
                    lx2_min,
                    lx3_min,
                    nx1_mb,
                    nx2_mb,
                    nx3_mb,
                )
        else:
            # pure python stitching
            for mb_id in range(filedata["n_mbs"]):
                lx1_idx = filedata["mb_logical"][mb_id][0] - lx1_min
                lx2_idx = filedata["mb_logical"][mb_id][1] - lx2_min
                lx3_idx = filedata["mb_logical"][mb_id][2] - lx3_min

                i_start = lx1_idx * nx1_mb
                j_start = lx2_idx * nx2_mb
                k_start = lx3_idx * nx3_mb

                i_end = i_start + nx1_mb
                j_end = j_start + nx2_mb
                k_end = k_start + nx3_mb

                for name in vars_to_use:
                    block = filedata["mb_data"][name][mb_id]  # (k, j, i)
                    stacked[name][i_start:i_end, j_start:j_end, k_start:k_end] = np.transpose(
                        block, (2, 1, 0)
                    )

        return {
            "data": stacked,
            "filedata": filedata,
        }

    def read(self, what: str):
        """
        Lazily read specified field(s) into the object.
        what: 'vel', 'dens', 'press', 'mag', 'cur'
        """
        if what in ("vel", "dens", "press"):
            needed = []
            if what in ("dens", "press") and "dens" not in self._loaded_w:
                needed.append("dens")
            if what == "press" and "eint" not in self._loaded_w:
                needed.append("eint")
            if what == "vel":
                for vname in ("velx", "vely", "velz"):
                    if vname not in self._loaded_w:
                        needed.append(vname)

            if needed:
                w_res = self._convert_and_stack(self.w_binary_fname, variables=needed)
                if self.meta_w is None:
                    self.meta_w = w_res["filedata"]
                self._loaded_w.update(needed)
                self._cache_w.update(w_res["data"])

            if what in ("dens", "press") and self.dens is None and "dens" in self._cache_w:
                self.dens = self._cache_w["dens"][None, ...]
            if what == "vel" and self.vel is None:
                if all(v in self._cache_w for v in ("velx", "vely", "velz")):
                    self.vel = np.stack(
                        [
                            self._cache_w["velx"],
                            self._cache_w["vely"],
                            self._cache_w["velz"],
                        ],
                        axis=0,
                    )
            if what == "press" and self.press is None:
                if self.dens is not None and "eint" in self._cache_w:
                    gamma = self._get_param_from_header(self.meta_w, "mhd", "gamma")
                    self.press = (gamma - 1.0) * self.dens * self._cache_w["eint"][None, ...]
        if what == "mag":
            if self.meta_b is None or self.mag is None:
                b_res = self._convert_and_stack(self.b_binary_fname, variables=["bcc1", "bcc2", "bcc3"])
                self.meta_b = b_res["filedata"]
                self.mag = np.stack(
                    [
                        b_res["data"]["bcc1"],
                        b_res["data"]["bcc2"],
                        b_res["data"]["bcc3"],
                    ],
                    axis=0,
                )
        if what == "cur":
            c_res = self._convert_and_stack(self.cur_binary_fname)
            names = [n for n in ("curx", "cury", "curz") if n in c_res["data"]]
            if not names:
                names = list(c_res["data"].keys())
            self.cur = np.stack([c_res["data"][n] for n in names], axis=0)

    @staticmethod
    def _stitch_blocks_numba(out_arr, blocks, mb_logical, lx1_min, lx2_min, lx3_min,
                             nx1_mb, nx2_mb, nx3_mb):
        """Numba-accelerated stitching of blocks into out_arr (x,y,z order)."""
        if not _NUMBA_AVAILABLE:
            raise RuntimeError("Numba not available")
        _stitch_blocks_numba_impl(out_arr, blocks, mb_logical, lx1_min, lx2_min, lx3_min,
                                  nx1_mb, nx2_mb, nx3_mb)


if _NUMBA_AVAILABLE:
    _stitch_sig_f32 = types.void(
        types.float32[:, :, :],          # out_arr
        types.float32[:, :, :, :],       # blocks
        types.int64[:, :],               # mb_logical
        types.int64, types.int64, types.int64,  # lx1_min, lx2_min, lx3_min
        types.int64, types.int64, types.int64,  # nx1_mb, nx2_mb, nx3_mb
    )
    _stitch_sig_f64 = types.void(
        types.float64[:, :, :],
        types.float64[:, :, :, :],
        types.int64[:, :],
        types.int64, types.int64, types.int64,
        types.int64, types.int64, types.int64,
    )

    @njit([_stitch_sig_f32, _stitch_sig_f64], parallel=True, cache=False, fastmath=False)
    def _stitch_blocks_numba_impl(out_arr, blocks, mb_logical, lx1_min, lx2_min, lx3_min,
                                  nx1_mb, nx2_mb, nx3_mb):
        nmb = blocks.shape[0]
        for mb_id in prange(nmb):
            lx1_idx = mb_logical[mb_id, 0] - lx1_min
            lx2_idx = mb_logical[mb_id, 1] - lx2_min
            lx3_idx = mb_logical[mb_id, 2] - lx3_min

            i_start = lx1_idx * nx1_mb
            j_start = lx2_idx * nx2_mb
            k_start = lx3_idx * nx3_mb

            for kk in range(nx3_mb):
                for jj in range(nx2_mb):
                    for ii in range(nx1_mb):
                        out_arr[i_start + ii, j_start + jj, k_start + kk] = blocks[mb_id, kk, jj, ii]


def load_athenak(file_number: int, basename: str = "CS", data_dir: str = ".") -> AthenaKBinData:
    """Helper to construct an AthenaKBinData object."""
    return AthenaKBinData(file_number=file_number, basename=basename, data_dir=data_dir)
