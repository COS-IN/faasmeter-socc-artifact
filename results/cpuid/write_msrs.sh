#!/bin/bash 

echo "Writing 0x38D - IA32_FIXED_CTR_CTRL -> $(sudo wrmsr 0x38D 0x33 -a)"
echo "        Control MSR for Fixed-Function Performance Counters 0-4"
echo " "

