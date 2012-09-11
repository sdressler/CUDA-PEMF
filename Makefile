TARGET = cudapmeter

OBJ		= power.hpp threads.hpp
CXX		= g++
CXXFLAGS = -g -O3 -I/opt/tdk/1.285.1/include -lnvidia-ml -pthread -shared -fPIC

$(TARGET) :
	$(CXX) $(CXXFLAGS) -o lib$(TARGET).so $(OBJ)

clean:
	rm -f *.gch
	rm -f lib$(TARGET).so
