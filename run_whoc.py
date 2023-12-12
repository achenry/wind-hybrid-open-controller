import multiprocessing as mp

def run_zmq():
    connect_zmq = True
    s = turbine_zmq_server(network_address="tcp://*:5555", timeout=10.0, verbose=True)
    while connect_zmq:
        #  Get latest measurements from ROSCO
        measurements = s.get_measurements()

        # Decide new control input based on measurements
        current_time = measurements['Time']
        if current_time <= 10.0:
            yaw_setpoint = 0.0
        else:
            yaw_setpoint = 20.0

            # Send new setpoints back to ROSCO
        s.send_setpoints(nacelleHeading=yaw_setpoint)

        if measurements['iStatus'] == -1:
            connect_zmq = False
            s._disconnect()

if __name__ == "__main__":
    p1 = mp.Process(target=run_zmq)
    p1.start()
    p2 = mp.Process(target=sim_rosco)
    p2.start()
    p1.join()
    p2.join()