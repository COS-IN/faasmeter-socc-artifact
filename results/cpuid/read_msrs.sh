#!/bin/sh


echo "Reading 0x38C - IA32_PERF_GLOBAL_CTRL -> $(sudo rdmsr 0x38C)"
echo " "
echo " "

echo "Reading 0x38E - IA32_PERF_GLOBAL_STATUS -> $(sudo rdmsr 0x38E)"
echo " "
echo " "

echo "Reading 0x390 - IA32_PERF_GLOBAL_OVF_CTRL -> $(sudo rdmsr 0x390)"
echo " "
echo " "

echo "Reading 0x38D - IA32_FIXED_CTR_CTRL -> $(sudo rdmsr 0x38D)"
echo "        Control MSR for Fixed-Function Performance Counters 0-4"
echo " "

echo "Reading 0x309 - IA32_FIXED_CTR0 -> $(sudo rdmsr 0x309)"
echo "        Fixed-Function Performance Counter 0 (R/W): Counts Instr_Retired.Any."
echo " "

echo "Reading 0x3OA - IA32_FIXED_CTR1 -> $(sudo rdmsr 0x30A)"
echo "        Fixed-Function Performance Counter 1 (R/W): Counts CPU_CLK_Unhalted.Core."
echo " "

echo "Reading 0x3OB - IA32_FIXED_CTR2 -> $(sudo rdmsr 0x30B)"
echo "        Fixed-Function Performance Counter 2 (R/W): Counts CPU_CLK_Unhalted.Ref."
echo " "


echo "Reading 0x606 - MSR_RAPL_POWER_UNIT -> $(sudo rdmsr 0x606)"
echo "        Unit multiplier to use with rapl msrs."
echo " "

# 639H energy status MSR_PP0_ENERGY_STATUS
# 64DH MSR_PLATFORM_ENERGY_COUNTER


