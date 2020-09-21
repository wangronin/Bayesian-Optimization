#/usr/bin/make


OBJ=CovFuns.o
TARGET=libcov.so
CC=gcc

$(TARGET):$(OBJ)
	$(CC) -shared -o $@ $^

CovFuns.o:CovFuns.c
	$(CC) -fPIC -Wall -c $< -o $@ 

.PHONY:clean

clean:
	rm -rf $(TARGET) $(OBJ)
