#ifndef TYPES_H
#define TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cstdint>
#include <format>
#include <string>
#include <iostream>

#define NSPEEDS         9
#define NUM_THREADS     28

typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  float  density;       /* density per cell */
  float  viscosity;     /* kinematic viscosity of fluid */
  float  velocity;      /* inlet velocity */
  int    type;          /* inlet type */
  float  omega;         /* relaxation parameter */
} t_param;

/* struct to hold the distribution of different speeds */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

class speed_identifier {
    private:
        std::uint32_t _data;

    public:
        speed_identifier() : _data{0} {}
        speed_identifier(const speed_identifier&) = default;
        speed_identifier(const float& v) : _data{reinterpret_cast<const std::uint32_t&>(v)} {}
        speed_identifier(const std::uint32_t v) : _data{v} {}
        operator float() const { return reinterpret_cast<const float&>(_data); }
        operator std::uint32_t() const { return _data; }

        void set_speed(unsigned int speed) {
            speed &= 0x0000000fU;
            _data &= 0xfffffff0U;
            _data |= speed;
        }

        unsigned int get_speed() const {
            return _data & 0x0000000fU;
        }

        void set_x(unsigned int x) {
            x &= 0x00003fffU;
            _data &= 0xfffc000fU;
            _data |= x << 4;
        }

        unsigned int get_x() const {
            return (_data & 0x0003fff0U) >> 4;
        }

        void set_y(unsigned int y) {
            y &= 0x00003fffU;
            _data &= 0x0003ffff;
            _data |= y << 18;
        }

        unsigned int get_y() const {
            return _data >> 18;
        }

        operator std::string() const {
            return std::format("from speed:{}, x:{}, y:{}", get_speed(), get_x(), get_y());
        }
};

#endif