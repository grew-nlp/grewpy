#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int get_shared_value(char *shared){
    for(int j = 0 ; j < 8 ; ++j)
        if (shared[j]) return j;
    return -1;
}
int main(void){
    // assume file exists
    int fd = -1;
    if ((fd = open("channel_out.txt", O_RDWR, 0)) == -1){
        printf("unable to open pods.txt\n");
        return 0;
    }
    int fanswer = open("channel_in.txt", O_RDWR, 0);
    // open the file in shared memory
    char *shared = (char *)mmap(NULL, 9, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    char *shared_answer = (char *)mmap(NULL, 9, PROT_READ | PROT_WRITE, MAP_SHARED, fanswer, 0);

    // periodically read the file contents
    while (true){
        int x = ((int *) shared)[0];
        printf("%d \n", x);
        if (x > 8){
            shared_answer[0] = '\x01';
            msync(shared, 128, MS_SYNC);

        }
        else{
            if (x <= 2){
                if (shared_answer[0]){
                    printf("--%d--%d--\n", x, shared_answer[0]);
                    shared_answer[0] = '\x00';
                    msync(shared, 1, MS_SYNC);
                }
            }
        } 
        sleep(1);
    }

    return 0;
}