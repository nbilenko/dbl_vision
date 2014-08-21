CXX = clang++
CXXFLAGS = -O2 --std=c++11

# Install prefix for opencv
OCV = $(CV)
LDFLAGS = -L$(OCV)/lib
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc

CXXFLAGS += -I$(OCV)/include

O = obj
B = bin

OBJ = $O/retinotopy.o

all: $O $B $B/retinotopy.x

$B/retinotopy.x: $(OBJ)
	$(CXX) $(LDFLAGS) $(OBJ) -o $@ $(LDLIBS)

$O/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$O $B:
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $O

.PHONY: distclean
distclean:
	rm -rf $O $B
