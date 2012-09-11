TARGET = memcpy

OBJ = memcpy.cu

all:
	nvcc -O0 -DINT -DADD -maxrregcount=21 -arch=sm_20 -use_fast_math -DTHREADS_PER_BLOCK=1024 -I/opt/tdk/1.285.1/include -lnvidia-ml -o $(TARGET) $(OBJ)

run:
	./$(TARGET) | ../driveGnuPlots.pl 1 $(X) CUDA 1000x500+0+0

clean:
	rm -f $(TARGET)
