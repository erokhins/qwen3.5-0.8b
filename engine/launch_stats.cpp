// The counters `engine/launch_stats.h` documents. Read that header first: it owns WHY the counters
// exist (bit-identical output cannot tell a batched wave from a per-row wave) and what is in scope.
#include "engine/launch_stats.h"

#include <sstream>

namespace engine::launchstats {

void Family::record(int rows) {
    if (rows < 1) return;
    ++calls;
    rows_total += rows;
    if (rows > 1) ++calls_batched;
    if (rows > rows_max) rows_max = rows;
}

std::string Family::describe() const {
    std::ostringstream os;
    os << "calls=" << calls << " calls_batched=" << calls_batched
       << " rows_total=" << rows_total << " rows_max=" << rows_max;
    return os.str();
}

Family& glue() {
    static Family f;
    return f;
}
Family& mlp_gateup() {
    static Family f;
    return f;
}
Family& mlp_down() {
    static Family f;
    return f;
}

std::string describe_run() {
    std::ostringstream os;
    os << "LAUNCHSTAT family=glue " << glue().describe() << "\n"
       << "LAUNCHSTAT family=mlp_gateup " << mlp_gateup().describe() << "\n"
       << "LAUNCHSTAT family=mlp_down " << mlp_down().describe();
    return os.str();
}

}  // namespace engine::launchstats
