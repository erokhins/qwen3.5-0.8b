// Compile-only TU for the MERGED-form census (D46 / bench/performance-status.md §18): the two forms the
// `rec2c` screen never compared, in ONE listing, so the census can bill the merge itself.
//
//   rec_kernel           the pre-unroll4 shipped spelling — the census's MODE-0 control row
//   rec0_kernel          MODE 0, `rec_kernel` copied verbatim (the in-TU control)
//   rec_unroll4_kernel   the SHIPPING spelling (engine/gdn_rec_unroll4_kernels.h, D33 default)
//   recx_shape_kernel    the prototype's own twin of it, kept as the third witness
//   rec2c_kernel         MODE 2c: the fused prologue on the PRE-unroll4 walks (the screened form)
//   rec2c4_kernel        MODE 2c4: the fused prologue MERGED with the shipping unroll4 walks
//
// Nothing here is ever launched; this TU exists so `isa_census.py` can answer "does the merge keep both?"
// without a device window. It needs `-I.` because the engine header names its op header as
// `prototype/gdn_decode/...`, which resolves only from the repo root (`build.sh` passes the same `-I.`).
#include "gdn_rec_isa_kernels.h"
#include "engine/gdn_rec_unroll4_kernels.h"
