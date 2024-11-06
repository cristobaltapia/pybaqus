# This makefile is used to generate and run the input files for different tests.
#
# To run all the tests:
# > make all
#

SHELL:=/bin/bash

INP_FILES := $(wildcard input/*.inp)
FIL_FILES := $(subst input,fil,$(subst .inp,.fil,$(INP_FILES)))

# Define the element types used to generate corresponding *.inp files
ELEM_TYPES_QUAD_4 := CPE4 CPE4H CPS4 CPS4I CPS4R
ELEM_TYPES_TRI_3 := CPE3 CPE3H CPS3
ELEM_TYPES_QUAD_8 := CPS8

# Generate paths for input files
INP_TARGETS_QUAD := $(addprefix input/quad_,$(addsuffix .inp,$(ELEM_TYPES_QUAD_4)))
INP_TARGETS_TRI := $(addprefix input/tri_,$(addsuffix .inp,$(ELEM_TYPES_TRI_3)))

INP_TARGETS := $(INP_TARGETS_TRI) $(INP_TARGETS_QUAD)

.PHONY: all clear $(ELEM_TYPES_QUAD_4)

all: $(INP_TARGETS) $(FIL_FILES)

# This rule runs each input file in the directory
fil/%.fil: input/%.inp
	@echo "> Starting Abaqus job $*..."
	@rm -f fil/$*.fil 2>/dev/null
	@abq2023 job=$* input=$< || { \
		echo "  Abaqus failed to start"; \
		exit 1; \
	}
	@until [ -f $*.sta ]; do \
		if ! pgrep -f "abq2023.*job=$*" >/dev/null; then \
			echo "  Process terminated unexpectedly before .sta file was created"; \
			[ -f $*.msg ] && { echo "Last few lines of message file:"; tail -n 20 $*.msg; }; \
			exit 1; \
		fi; \
		sleep 2; \
	done
	@# Check that the analysis has finished correctly
	@until grep -q "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" $*.sta 2>/dev/null; do \
		if ! pgrep -f "abq2023.*job=$*" >/dev/null; then \
			echo "  Process terminated unexpectedly during analysis"; \
			echo "  Last few lines of status file:"; \
			tail -n 20 $*.sta 2>/dev/null || true; \
			[ -f $*.msg ] && { echo "  Last few lines of message file:"; tail -n 20 $*.msg; }; \
			exit 1; \
		fi; \
		sleep 2; \
	done
	@# Check that the lock file is gone
	@until [ ! -f $*.lck ]; do \
		sleep 1; \
	done
	@echo "  Analysis for $* completed successfully"
	@rm -f $*.dat $*.odb $*.com $*.msg $*.sta $*.log $*.prt
	@mv $*.fil fil/$*.fil

# Generate individual input files per element type with quad shape
input/quad_%.inp: input/templates/elem_quad.inp
	@mkdir -p input
	@echo "Generating input file for element type $*..."
	@sed -e 's/{ELEM_TYPE}/$*/g' $< > $@

# Generate individual input files per element type with triangular shape
input/tri_%.inp: input/templates/elem_tri.inp
	@mkdir -p input
	@echo "Generating input file for element type $*..."
	@sed -e 's/{ELEM_TYPE}/$*/g' $< > $@

clear:
	rm fil/*.fil
