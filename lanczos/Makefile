# Makefile layout originally developed by Bill Lynch
SOURCES     := $(sort $(wildcard src/*.cpp src/*.cu src/*.cc src/*.c))
OBJECTS     := $(sort $(patsubst src/%, obj/%.o, $(SOURCES)))
DEPENDS     := $(sort $(patsubst src/%, obj/%.d, $(SOURCES)))

CFLAGS      += -std=gnu99 -I./inc -I/usr/global/cuda/5.0/include -I/usr/global/openmpi/022513/gnu/include
LDFLAGS     += -L/usr/global/cuda/5.0/lib64 -lcudart -lcuda -lcublas -lmpi -L/usr/global/openmpi/022513/gnu/lib
LDFLAGS     += -L/usr/global/intel/mkl/2011.4/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_sequential -lm -fopenmp
CUFLAGS     += -arch=sm_20 -I./inc -I/usr/global/openmpi/022513/gnu/lib
CUFLAGS     += --ptxas-options=-v

NVCC        := nvcc -g -G 
CXX         := mpicc -g

bin/lan.x: $(OBJECTS)
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
