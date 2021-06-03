#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Int8
import RPi.GPIO as GPIO

def decodeMsg(msg):
    """
    Decodes Message (Int8) into the PWM (% duty cycle) of the right and left motor signals 
    based on the following encoding:
        b'up dn lt rt
    Example: 1001 (9) => up & right
    """
    duty_cycle_r = 0
    duty_cycle_l = 0
    d_duty = 30 # step of duty cycle [%]

    if msg & (1<<3):
        duty_cycle_l += d_duty
        duty_cycle_r += d_duty
    elif msg & (1<<2): # Going fwd and bwd at the same time makes no sense
        duty_cycle_l -= d_duty
        duty_cycle_r -= d_duty

    if msg & (1<<1):
        duty_cycle_l -= d_duty
        duty_cycle_r += d_duty
    elif msg & 1: # Turning left and right at the same time makes no sense
        duty_cycle_l += d_duty
        duty_cycle_r -= d_duty

    return (duty_cycle_r, duty_cycle_l)


class DriverNode(object):
    def __init__(self):
        rospy.init_node('driver_node')
        rospy.Subscriber('ottobot/key_teleop', Int8, self.callback)
        
        # Pin setup
        # IN1 HIGH & IN2 LOW -> RIGHT FWD
        # IN1 LOW & IN2 HIGH -> RIGHT BWD
        # IN3 LOW & IN4 HIGH -> LEFT FWD
        # IN3 HIGH & IN4 LOW -> LEFT BWD

        GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(IN4, GPIO.OUT, initial=GPIO.HIGH)

        GPIO.setup(ENA, GPIO.OUT)
        GPIO.setup(ENB, GPIO.OUT)

        self.motor_r_pwm = GPIO.PWM(ENA, 100)
        self.motor_l_pwm = GPIO.PWM(ENB, 100)

        self.r_duty = 30
        self.l_duty = 30

        self.last_r_duty = None
        self.last_l_duty = None

        rospy.loginfo('Starting PWMs')
        self.motor_l_pwm.start(self.l_duty)
        self.motor_r_pwm.start(self.r_duty)

    def run(self):
        rospy.spin()

    def callback(self, data):
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        duty_cycle_r, duty_cycle_l = decodeMsg(data.data)
        # A negative value in duty means that the motor is going backwards
        
        if duty_cycle_r != self.r_duty:
            if duty_cycle_r == 0:
                # Right motor not moving
                GPIO.output(IN1, GPIO.LOW)
                GPIO.output(IN2, GPIO.LOW)
            elif duty_cycle_r < 0:
                # Right motor moving backwards
                GPIO.output(IN1, GPIO.LOW)
                GPIO.output(IN2, GPIO.HIGH)
            else:
                # Right motor moving forwards
                GPIO.output(IN1, GPIO.HIGH)
                GPIO.output(IN2, GPIO.LOW)
            
            self.r_duty = duty_cycle_r
            self.motor_r_pwm.ChangeDutyCycle(abs(self.r_duty))

        if duty_cycle_l != self.l_duty:
            if duty_cycle_l == 0:
                # Right motor not moving
                GPIO.output(IN3, GPIO.LOW)
                GPIO.output(IN4, GPIO.LOW)
            elif duty_cycle_l < 0:
                # Right motor moving backwards
                GPIO.output(IN3, GPIO.HIGH)
                GPIO.output(IN4, GPIO.LOW)
            else:
                # Right motor moving forwards
                GPIO.output(IN3, GPIO.LOW)
                GPIO.output(IN4, GPIO.HIGH)
            
            self.l_duty = duty_cycle_l
            self.motor_l_pwm.ChangeDutyCycle(abs(self.l_duty))

        rospy.loginfo("Right : {} || Left: {}".format(duty_cycle_r, duty_cycle_l))
 
if __name__ == '__main__':
    
    # Pin assignment
    ENA = 33
    IN1 = 36
    IN2 = 31

    ENB = 32
    IN3 = 35
    IN4 = 37
    GPIO.setmode(GPIO.BOARD)

    driver_node = DriverNode()
    try:
        driver_node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        GPIO.cleanup()