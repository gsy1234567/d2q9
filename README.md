# d2q9

## Stream and Boundary
- We detect stream and boundary law by taking for an example with the following parameters file.

```
nx: 4
ny: 4
iters: 1
density: 1
viscosity: 0.02
velocity: 0.05
type: 1
```
- The following tables show the result after a stream-boundary step. Eech term is a three-elements tuple `(src_x, src_y, src_speed)`.
### Speed0
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
| y=3 | (0, 3, 0)| (1, 3, 0)| (2, 3, 0)| (2, 3, 0) |
| y=2 | (0, 2, 0)| (1, 2, 0)| (2, 2, 0)| (2, 2, 0) |
| y=1 | (0, 1, 0)| (1, 1, 0)| (2, 1, 0)| (2, 1, 0) |
| y=0 | (0, 0, 0)| (1, 0, 0)| (2, 0, 0)| (2, 0, 0) |
### The law of speed0
```
if(x_idx < params.nx - 1) {
    dev_speeds_dst.at(x_idx,y_idx,0)=tmp[0];
    if(x_idx == params.nx - 2) {
        dev_speeds_dst.at(x_idx+1,y_idx,0)=tmp[0];
    }
}
```
### Speed1 
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
|y=3|N/A| (0, 3, 1)| (1, 3, 1)| (1, 3, 1) |
|y=2|N/A| (0, 2, 1)| (1, 2, 1)| (1, 2, 1) |
|y=1|N/A| (0, 1, 1)| (1, 1, 1)| (1, 1, 1) |
|y=0|N/A| (0, 0, 1)| (1, 0, 1)| (1, 0, 1) |
### The law of speed1
```
if(x_idx < params.nx - 2) {
    dev_speeds_dst.at(x_idx+1,y_idx,1) = tmp[1];
    if(x_idx == params.nx-3) {
        dev_speeds_dst.at(x_idx+2,y_idx,1) = tmp[1];
    }
}
```
### Speed2
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
|y=3|(0, 2, 2)|(1, 2, 2)|(2, 2, 2)|(2, 2, 2)|
|y=2|(0, 1, 2)|(1, 1, 2)|(2, 1, 2)|(2, 1, 2)|
|y=1|(0, 0, 2)|(1, 0, 2)|(2, 0, 2)|(2, 0, 2)|
|y=0|(0, 0, 4)|(1, 0, 4)|(2, 0, 4)|(2, 0, 4)|
### The law of speed2
```
if(x_idx < params.nx-1 && y_idx < params.ny-1) {
    dev_speeds_dst.at(x_idx,y_idx+1,2) = tmp[2];
    if(y_idx == 0) {
        dev_speeds_dst.at(x_idx,0,2) = tmp[4];
        if(x_idx == params.nx-2) {
            dev_speeds_dst.at(x_idx+1,0,2) = tmp[4]
        }
    }
    if(x_idx == params.nx-2) {
        dev_speeds_dst.at(x_idx+1,y_idx+1,2) = tmp[2];
    }
}
```
### Speed3
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
|y=3|(1, 3, 3)|(2, 3, 3)|(3, 3, 3)|(3, 3, 3)|
|y=2|(1, 2, 3)|(2, 2, 3)|(3, 2, 3)|(3, 2, 3)|
|y=1|(1, 1, 3)|(2, 1, 3)|(3, 1, 3)|(3, 1, 3)|
|y=0|(1, 0, 3)|(2, 0, 3)|(3, 0, 3)|(3, 0, 3)|
### The law of speed3
```
if(x_idx > 0) {
    dev_speeds_dst.at(x_idx-1,y_idx,3) = tmp[3];  
    if(x_idx == params.nx-1) {
        dev_speeds_dst.at(x_idx,y_idx,3) = tmp[3];
    } 
}
```
### Speed4
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
|y=3|(0, 3, 2)|(1, 3, 2)|(2, 3, 2)|(2, 3, 2)|
|y=2|(0, 3, 4)|(1, 3, 4)|(2, 3, 4)|(2, 3, 4)|
|y=1|(0, 2, 4)|(1, 2, 4)|(2, 2, 4)|(2, 2, 4)|
|y=0|(0, 1, 4)|(1, 1, 4)|(2, 1, 4)|(2, 1, 4)|
### The law of speed4
```
if(x_idx < params.nx-1 && y_idx > 0) {
    dev_speeds_dst.at(x_idx,y_idx-1,4) = tmp[4];
    if(y_idx == params.ny-1) {
        dev_speeds_dst.at(x_idx,y_idx,4) = tmp[2];
        if(x_idx == params.nx-2) {
            dev_speeds_dst.at(x_idx+1,y_idx,4) = tmp[2];
        }
    }
    if(x_idx == params.nx-2) {
        dev_speeds_dst.at(x_idx+1,y_idx,4) = tmp[4];
    }
}
```
### Speed5
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
|y=3|N/A|(0, 2, 5)|(1, 2, 5)|(1, 2, 5)|
|y=2|N/A|(0, 1, 5)|(1, 1, 5)|(1, 1, 5)|
|y=1|N/A|(0, 0, 5)|(1, 0, 5)|(1, 0, 5)|
|y=0|N/A|(1, 0, 7)|(2, 0, 7)|(2, 0, 7)|
### The law of speed5
```
if(y_idx==0 && x_idx>0 && x_idx<params.nx-1) {
    dev_speeds_dst.at(x_idx,0,5) = tmp[7];
    if(x_idx==params.nx-2) {
        dev_speeds_dst.at(x_idx+1,0,5) = tmp[7];
    }
}
if(x_idx<params.nx-2 && y_idx<params.ny-1) {
    dev_speeds_dst.at(x_idx+1,y_idx+1,5) = tmp[5];
    if(x_idx==params.nx-3) {
        dev_speeds_dst.at(x_idx+2,y_idx+1,5) = tmp[5];
    } 
}
```
### Speed6
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
|y=3|(1, 2, 6)|(2, 2, 6)|(3, 2, 6)|(3, 2, 6)|
|y=2|(1, 1, 6)|(2, 1, 6)|(3, 1, 6)|(3, 1, 6)|
|y=1|(1, 0, 6)|(2, 0, 6)|(3, 0, 6)|(3, 0, 6)|
|y=0|(0, 0, 8)|(1, 0, 8)|(2, 0, 8)|(2, 0, 8)|
### The law of speed6
```
if(y_idx==0 && x_idx<params.nx-1) {
    dev_speeds_dst.at(x_idx,0,6) = tmp[8];
    if(x_idx == params.nx-2) {
        dev_speeds_dst.at(x_idx+1,0,6)=tmp[8];
    }
}
if(x_idx>0 && y_idx<params.ny-1) {
    dev_speeds_dst.at(x_idx-1,y_idx+1,6)=tmp[6];
    if(x_idx == params.nx-1) {
        dev_speeds_dst.at(x_idx-1,y_idx,6)=tmp[6];
    }
}
```
### Speed7
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
|y=3|(0, 3, 5)|(1, 3, 5)|(2, 3, 5)|(2, 3, 5)|
|y=2|(1, 3, 7)|(2, 3, 7)|(3, 3, 7)|(3, 3, 7)|
|y=1|(1, 2, 7)|(2, 2, 7)|(3, 2, 7)|(3, 2, 7)|
|y=0|(1, 1, 7)|(2, 1, 7)|(3, 1, 7)|(3, 1, 7)|
### The law of speed7
```
if(y_idx==params.ny-1 && x_idx<params.nx-1) {
    dev_speeds_dst.at(x_idx,y_idx,7)=tmp[5];
    if(x_idx == params.nx-2) {
        dev_speeds_dst.at(x_idx+1,y_idx,7)=tmp[5];
    }
}
if(x_idx>0 && y_idx>0) {
    dev_speeds_dst.at(x_idx-1,y_idx-1,7)=tmp[7];
    if(x_idx == params.nx-1) {
        dev_speeds_dst.at(x_idx,y_idx-1,7)=tmp[7];
    }
}
```
### Speed8
||x=0|x=1|x=2|x=3|
|-|-|-|-|-|
|y=3|N/A|(1, 3, 6)|(2, 3, 6)|(2, 3, 6)|
|y=2|N/A|(0, 3, 8)|(1, 3, 8)|(1, 3, 8)|
|y=1|N/A|(0, 2, 8)|(1, 2, 8)|(1, 2, 8)|
|y=0|N/A|(0, 1, 8)|(1, 1, 8)|(1, 1, 8)|
### The law of speed8
```
if(y_idx==params.ny-1 && x_idx>0 && x_idx<params.nx-1) {
    dev_speeds_dst.at(x_idx,y_idx,8)=tmp[6];
    if(x_idx == params.nx-2) {
        dev_speeds_dst.at(x_idx+1,y_idx,8)=tmp[6];
    }
}
if(x_idx<params.nx-2 && y_idx>0) {
   dev_speeds_dst.at(x_idx+1,y_idx-1,8)=tmp[8];
   if(x_idx == params.nx-3) {
        dev_speeds_dst.at(x_idx+2,y_idx-1,8)=tmp[8];
   }
}
```
- We can combine streaming step and three sub-steps of boundary (top wall, bottom wall and right wall) into one in order to reduce the numbers of reading and writing global memory. The practical way is writing the correct speed to the correct location.

