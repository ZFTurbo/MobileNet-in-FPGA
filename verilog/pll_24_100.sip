set_global_assignment -entity "pll_24_100" -library "lib_pll_24_100" -name IP_TOOL_NAME "altera_pll"
set_global_assignment -entity "pll_24_100" -library "lib_pll_24_100" -name IP_TOOL_VERSION "18.0"
set_global_assignment -entity "pll_24_100" -library "lib_pll_24_100" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_pll_24_100" -name SPD_FILE [file join $::quartus(sip_path) "pll_24_100.spd"]

set_global_assignment -library "lib_pll_24_100" -name MISC_FILE [file join $::quartus(sip_path) "pll_24_100_sim/pll_24_100.vo"]
