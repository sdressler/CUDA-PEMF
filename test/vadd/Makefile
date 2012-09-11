TARGET = vadd

OBJ = $(TARGET).cu

all:
	nvcc -arch=sm_20 -use_fast_math -I/opt/tdk/1.285.1/include -lnvidia-ml -o $(TARGET) $(OBJ) -Xptxas -dlcm=ca

ptx:
	nvcc -arch=sm_20 -use_fast_math -I/opt/tdk/1.285.1/include -lnvidia-ml $(OBJ) -ptx -Xptxas -dlcm=ca

run:
	./$(TARGET) | ../driveGnuPlots.pl 2 $(X) $(X) Power Utilization 1300x300+0+0 1300x300+0+320

clean:
	rm -f $(TARGET) *.ptx
