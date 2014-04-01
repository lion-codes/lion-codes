# Makefile layout originally developed by Bill Lynch
SOURCES     := $(sort $(wildcard src/*.cpp src/*.cu src/*.cc src/*.c))
OBJECTS     := $(sort $(patsubst src/%, obj/%.o, $(SOURCES)))
DEPENDS     := $(sort $(patsubst src/%, obj/%.d, $(SOURCES)))

CPPFLAGS      += -I./inc -I/usr/local/cuda-5.5/include -I/gpfs/apps/cuda-rhel6/cuda/5.0/include/ -I/usr/global/boost/1.50.0/boost
LDFLAGS     += -L/usr/local/cuda-5.5/lib64 -lcudart -lcuda -lcusparse
LDFLAGS     += -L/usr/global/boost/1.50.0/libs/
CUFLAGS     += -arch=sm_30 -I./inc 
CUFLAGS     += --ptxas-options=-v 

NVCC        := nvcc -O3 -U__VERBOSE 
CXX         := g++ -O3 -U__VERBOSE

bin/fsolve.x: $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf bin obj

obj/%.cpp.o: src/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


obj/%.cu.o: src/%.cu
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c $< -o $@


obj/%.cpp.d: src/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM -MT "$(strip $(patsubst %.d, %.o, $@) $@)" $< -o $@


obj/%.cu.d: src/%.cu
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -M $< | sed -e '1 s:$*.o:obj/$*.cu.d obj/$*.cu.o:' -e 's:[^ ]*/ ::g' > $@ 


# Create system layout for Objects and binaries
ifneq ($(MAKECMDGOALS),clean)
$(shell mkdir -p bin obj)
-include $(DEPENDS)
endif
