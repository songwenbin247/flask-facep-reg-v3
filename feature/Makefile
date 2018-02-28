CFILES = feature.c

all: matrix

matrix: $(OBJS)
	gcc $(CFILES) -o libfeature.so -shared -fpic 

clean:
	rm -f *.o libmatrix.so

