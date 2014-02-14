# Makefile layout originally developed by Bill Lynch
SOURCES     := $(sort $(wildcard src/*.cpp src/*.cu src/*.cc src/*.c))
OBJECTS     := $(sort $(patsubst src/%, obj/%.o, $(SOURCES)))
DEPENDS     := $(sort $(patsubst src/%, obj/%.d, $(SOURCES)))

MKL=/usr/global/intel/mkl/11.0.0.079
MKL_LIB=$(MKL)/lib/intel64 -lmkl_rt
MKL_INC=$(MKL)/include

CUDA=/usr/global/cuda/5.5
CUDA_LIB=$(CUDA)/lib64
CUDA_INC=$(CUDA)/include

CFLAGS      += -std=gnu99 -I./inc -I$(CUDA_INC) -openmp -I$(MKL_INC)
LDFLAGS     += -L$(CUDA_LIB) -lcudart -lcuda -lcublas 
LDFLAGS     += -L$(MKL_LIB) -openmp 
CUFLAGS     += -gencode arch=compute_35,code=sm_35 -I./inc 
CUFLAGS     += --ptxas-options=-v 

NVCC        := nvcc -O3  #-D_QR_VERBOSE_
CXX         := icc -g -O3  -xHost #-D_QR_VERBOSE_

bin/qr.x: $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf bin obj

obj/%.cpp.o: src/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

obj/%.cc.o: src/%.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

obj/%.cu.o: src/%.cu
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c $< -o $@

obj/%.c.o: src/%.c
	$(CXX) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

obj/%.cpp.d: src/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM -MT "$(strip $(patsubst %.d, %.o, $@) $@)" $< -o $@

obj/%.cc.d: src/%.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM -MT "$(strip $(patsubst %.d, %.o, $@) $@)" $< -o $@

obj/%.cu.d: src/%.cu
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -M $< | sed -e '1 s:$*.o:obj/$*.cu.d obj/$*.cu.o:' -e 's:[^ ]*/ ::g' > $@ 

obj/%.c.d: src/%.c
	$(CXX) $(CPPFLAGS)  $(CFLAGS) -MM -MT "$(strip $(patsubst %.d, %.o, $@) $@)" $< -o $@

# Create system layout for Objects and binaries
ifneq ($(MAKECMDGOALS),clean)
$(shell mkdir -p bin obj)
-include $(DEPENDS)
endif
