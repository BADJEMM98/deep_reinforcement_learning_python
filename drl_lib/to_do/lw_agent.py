
class LineWorldAgent():
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y    

    def move_right(self,step:int):
        self.x+=step

    def move_left(self,step:int):
        self.x-=step
