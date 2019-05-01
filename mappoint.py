class Map_point():
    def __init__(self, id, coordinate, descriptor,pixel, norm=None):
        self.id = id
        self.position = coordinate
        self.descriptor=descriptor
        self.pixel=pixel
        self.norm = norm
        self.observed = 1
        self.corrected = 0

    def finetune(self, new_pos):
        self.position = new_pos
        self.corrected += 1
