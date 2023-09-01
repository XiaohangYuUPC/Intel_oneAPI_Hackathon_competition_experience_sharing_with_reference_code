CC=icc
CFLAGS=-Wall
LIBS=-lfftw3 -lmkl_rt -lm
TARGET=acceleration_fft

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

