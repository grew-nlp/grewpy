#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void){
    int fd = -1;
    if ((fd = open("channel_out.txt", O_RDWR, 0)) == -1)
        return 0;
    // open the file in shared memory
    char *shared = (char *)mmap(NULL, 9, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // periodically read the file contents
    while (true){
        int x = ((int *) shared)[0];
        printf("%d \n", x);
        if (x > 8){
            shared[0] = '\x00';
            msync(shared, 128, MS_SYNC);
        }
        sleep(1);
    }

    return 0;
}