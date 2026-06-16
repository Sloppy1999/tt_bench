# Turing Tumble Board Coordinate System

## Board Dimensions
- Width: 11 columns (x = 0 to 10)
- Height: 11 rows (y = 0 to 10)

## Coordinate Origin
- **(0, 0)** = Top-left corner of the playable grid
- **x** increases left to right (columns 0-10)
- **y** increases top to bottom (rows 0-10)

## Hopper Entry Mode

Official challenge JSONs use `"inward"` mode: a marble from the blue hopper
enters one column **right** of the hopper slot (x+1), and a red marble enters
one column **left** of its slot (x-1).

## Off-Board Components

### Ball Hoppers
Positioned above the grid at y = -1:
- Blue hopper: x = 2
- Red hopper: x = 8

### Trigger Levers (Catchers)
Positioned below the grid at y = 11:
- Left lever: x = 2
- Right lever: x = 8

## Component Type Vocabulary

| Type string | Description |
|---|---|
| `ramp_right` | Ramp directing ball down-right |
| `ramp_left` | Ramp directing ball down-left |
| `crossover` | Crossover piece |
| `bit` | Standard bit; requires `state` (0 or 1) |
| `gear_bit` | Gear bit; requires `state` (0 or 1) and `gear_group` |
| `gear` | Gear connector; requires `gear_group` |
| `interceptor` | Interceptor/catcher |
| `trigger` | Trigger lever; requires `hopper` ("blue" or "red") |
| `ball_hopper` | Ball hopper; requires `color` and `count` |

## Bit State Convention
- `state: 0` = Bit points to the left
- `state: 1` = Bit points to the right

## Gear Group Convention
- Gear bits and gears sharing the same `gear_group` integer are mechanically connected
- When one gear bit flips, all gears in the same group rotate
