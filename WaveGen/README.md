# Wave genereator control code

The wave generator has a [motor](https://www.omc-stepperonline.com/de/p-series-ip67-wasserdicht-nema-23-schrittmotor-5-0a-1-8nm-254-95oz-in-23ip67-20) and a [stepper driver](https://www.omc-stepperonline.com/digital-stepper-driver-1-0-4-2a-20-50vdc-for-nema-17-23-24-stepper-motor-dm542t).

# Arduino lib

The stepper driver is controlled by a arduino nano. The code is running a stepper [library](https://docs.arduino.cc/libraries/stepperdriver/).
For information on installing libraries, [here](http://www.arduino.cc/en/Guide/Libraries).

# Python code

| Argument   | Description                       | Type   |
|------------|-----------------------------------|--------|
| `angle`    | turns around a certain angle [deg]| int    |
| `rpm`      | Sets the rpm (default 60)         | Float  |
| `t`        | Limits the rotation time [s]      | Float  |
| `delay`    | Add a delayy before startin [s]   | Float  |

## rpm mode

If only passing rpm the motor will turn until t is reached with the given rpm.

```sh
pyhton3 WaveGen.py rpm 60 t 5 delay 0
```

## andgle mode

Sets the angle to turn and the rpm. The motor will turn until the angle is reached. if no rpm given then default 60 is used.

```sh
pyhton3 WaveGen.py angle 720 rpm 100 delay 0
```

# connection to pi on the WaveGen

connect to ```tank@tank.local``` after seting up a network conection




