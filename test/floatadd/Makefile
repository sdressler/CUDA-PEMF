TARGET = floatadd

OBJ = $(TARGET).cu

all:
	nvcc -O0 -arch=sm_20 -o $(TARGET) $(OBJ) -I/opt/tdk/1.285.1/include

ptx:
	nvcc -O0 -arch=sm_20 $(OBJ) -ptx

run:
	./$(TARGET) | ../driveGnuPlots.pl 2 $(X) $(X) Power Utilization 1300x300+0+0 1300x300+0+320

clean:
	rm -f $(TARGET) *.ptx
