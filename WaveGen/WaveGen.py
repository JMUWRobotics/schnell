import serial
import time
import sys

# Replace with the correct port and baud rate (e.g., 'COM3' for Windows or '/dev/ttyACM0' for Linux)
PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

rpm = 0.0
run_time = 0.0
angle = 0
delay = 0

def send_command(command):
    with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser:
        time.sleep(2)  # Give time for Arduino to reset

        ser.write((command + '\n').encode())
        print(f"Sent: {command}")
        while True:
            response = ser.readline().decode().strip()
            if(response != ""):
                print(f"Received: {response}")
                if(response == "Finished"):
                    break

def parser():

    global rpm, run_time, angle, delay
    
    if len(sys.argv) < 2:
        print("No inputs given, please use 't', 'rpm' or 'angle'")
        return False
   
    else:

       # get the arguments to the wrigth variabales
        for i in range(1, len(sys.argv)):

            if sys.argv[i] == 't':
                run_time = sys.argv[i + 1]
                print("run_time set to:", run_time)
                continue
            
            elif sys.argv[i] == 'rpm':
                rpm = sys.argv[i + 1] 
                print("rpm set to:", rpm)
                continue

            elif sys.argv[i] == 'angle':
                angle = sys.argv[i + 1]
                print("angle set to:", angle)
                continue
            elif sys.argv[i] == 'delay':
                delay = sys.argv[i + 1]
                print("start delay set to: ", delay)
                continue

            else:
                if sys.argv[i].isdigit():
                    continue

                else:
                    print("wrong input option, please use 't', 'rpm' or 'angle', %s is not valid" % sys.argv[i])
    
        return True


    

if __name__ == '__main__':

    is_cmd = parser()

    if not is_cmd:
        sys.exit(1)
    
    run_time = float(run_time) * 1000 # Convert to milliseconds  

    # time in ms
    cmd = str("t " + str(run_time) + " rpm " + str(rpm) + " a " + str(angle)) + " delay " + str(delay)

    print("Command to be sent: ", cmd)

    # Send the command to the Arduino
    send_command(cmd)
