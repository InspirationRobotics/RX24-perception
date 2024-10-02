

class Mission1:

    model1 = __path__
    model2 = __path__

    def __init__(self):
        self.end = False
        self.state = 0
        self.target = "narnia"

    def run(self, frame, position, og):
        print("Mission 1 is running")
        target_vector =[1,0]
        return dict("targete_vecotr"), ":chaneg to port camera to {model1}"

class Mission2:

    def __init__(self):
        self.state = 0
        self.target = "narnia"

    def run(self):
        print("Mission 2 is running")

class MissionHandler:
    def __init__(self, object : [Mission, Mission]):
        self.object = object
        load_cameras()
        load_occupancy_grid()

    def position_callback():
        self.position = position
    
    def frame_callback():
        lock:
            self.frame = frame

    def run(self):
        while self.object.end != True:
            output = self.object.run(frame, self.position, og)
            #check for relevant muission flaghs
            #  change settinsg accordingly
            # delete flags from output
            # send only relevant command flags
            output.encode()
            server.send(output)
        if output.model_flag = True:
  
        server.send("Mission Complete")


       