import pygame
import math
import sys
import random
import neat
import os
import numpy as np
import pickle
import gzip

pygame.init()

window = pygame.display.set_mode((500,500))
screen_rect=window.get_rect()

pygame.display.set_caption("Snake - By Joseph Goold")
font = pygame.font.Font('freesansbold.ttf', 24)
font2 = pygame.font.Font('freesansbold.ttf', 15)
text = font.render('Rip, you died. Press E to restart.', True, (255,0,0))
textRect = text.get_rect()  
        
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class snake:
    gens=0
    def __init__(self,pos,yspeed,xspeed,mainspeed,size,boost,color,timer):
        self.boost=boost
        self.pos=pos
        self.yspeed=yspeed
        self.xspeed=xspeed
        self.mainSpeed=mainspeed
        self.size=size
        self.color=color
        self.timer=timer
        self.prevPoints=[]
        self.dead=0
        self.pickTimer=0
    def pickupBoost(self):
        self.size+=1
        self.mainSpeed=clamp(self.mainSpeed-1,3,500)
        self.boost=(math.ceil(random.randint(1,490) / 10.0) * 10, math.ceil(random.randint(1,490) / 10.0) * 10)
w=10
h=10
timer=0

snakes=[]
nets=[]
ge=[]
clock=pygame.time.Clock()

def main(genomes,config):
    snake.gens+=1
    for _,g in genomes:
        net=neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        color=(random.randint(50,255),random.randint(50,255),random.randint(50,255))
        pos=(250,250)
        snakes.append(snake(pos,10,0,50,5,(240,240),color,0))
        snakes[-1].boost=(math.ceil(random.randint(1,490) / 10.0) * 10, math.ceil(random.randint(1,490) / 10.0) * 10)
        g.fitness=0
        ge.append(g)
    run=True
    while run:
        #Position controlling
        if(len(snakes)==0):
            break
        window.fill((0,0,0))
        tick=clock.tick()
        fitnesses=[]
        for ee in ge:
            fitnesses.append(ee.fitness)
        for x,snak in enumerate(snakes):
            if(snak.xspeed>0):
                input_layer = np.array([[1 if snak.pos[0]-10<=0 or (snak.pos[0]-10,snak.pos[1]) in snak.prevPoints else 0,1 if snak.pos[0]+10>=490 or (snak.pos[0]+10,snak.pos[1]) in snak.prevPoints else 0, 1 if snak.pos[1]+10>=490 or (snak.pos[0],snak.pos[1]+10) in snak.prevPoints else 0]])
            elif(snak.xspeed<0):
                input_layer = np.array([[1 if snak.pos[1]+10>=490 or (snak.pos[0],snak.pos[1]+10) in snak.prevPoints else 0,1 if snak.pos[0]-10<=0 or (snak.pos[0]-10,snak.pos[1]) in snak.prevPoints else 0, 1 if snak.pos[1]-10<=10 or (snak.pos[0],snak.pos[1]-10) in snak.prevPoints else 0]])
            elif(snak.yspeed>0):
                input_layer = np.array([[1 if snak.pos[0]+10>=490 or (snak.pos[0]+10,snak.pos[1]) in snak.prevPoints else 0,1 if snak.pos[1]+10>=490 or (snak.pos[0],snak.pos[1]+10) in snak.prevPoints else 0, 1 if snak.pos[0]-10<=10 or (snak.pos[0]-10,snak.pos[1]) in snak.prevPoints else 0]])
            elif(snak.yspeed<0):
                input_layer = np.array([[1 if snak.pos[0]-10<=10 or (snak.pos[0]-10,snak.pos[1]) in snak.prevPoints else 0,1 if snak.pos[1]-10<=0 or (snak.pos[0],snak.pos[1]-10) in snak.prevPoints else 0, 1 if snak.pos[0]+10>=490 or (snak.pos[0]+10,snak.pos[1]) in snak.prevPoints else 0]])
            distancex=snak.pos[0] - snak.boost[0]
            distancey=snak.pos[1] - snak.boost[1]
            outputs = nets[x].activate((snak.pos[0],snak.pos[1],snak.boost[0],snak.boost[1],input_layer[0][0],input_layer[0][1],input_layer[0][2],distancex,distancey))
            distance=(math.sqrt((snak.pos[0] - snak.boost[0])**2 + (snak.pos[1] - snak.boost[1])**2)) /100
            if(outputs[0] > 0.5 and snak.yspeed!=-10):
                snak.yspeed=10
                snak.xspeed=0
            elif(outputs[1] > 0.5 and snak.yspeed!=10):
                snak.yspeed=-10
                snak.xspeed=0
            elif(outputs[2] > 0.5 and snak.xspeed!=-10):
                snak.yspeed=0
                snak.xspeed=10
            elif(outputs[3] > 0.5 and snak.xspeed!=10):
                snak.yspeed=0
                snak.xspeed=-10
            
            if(ge[x].fitness >= max(fitnesses)):
                pygame.draw.rect(window, snak.color, (snak.pos[0],snak.pos[1],w,h))
                pygame.draw.rect(window, snak.color, (snak.boost[0],snak.boost[1],w,h))
                for e in snak.prevPoints:
                    pygame.draw.rect(window, snak.color, (e[0],e[1],w,h), 1)
            snak.timer+=tick
            snak.pickTimer+=tick
            if(snak.timer>=snak.mainSpeed and snak.dead==0):
                snak.timer=0
                newPoint = (snak.pos[0],snak.pos[1])
                snak.prevPoints.append(newPoint)
                if(len(snak.prevPoints) >= snak.size):
                    snak.prevPoints.pop(0)
                snak.pos=(snak.pos[0]+snak.xspeed,snak.pos[1]+snak.yspeed)
                if(snak.pos[0]<0 or snak.pos[0]>490):
                    snak.dead=1
                    ge[x].fitness-=2
                    snakes.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    continue
                if(snak.pos[1]<0 or snak.pos[1]>490):
                    snak.dead=1
                    ge[x].fitness-=2
                    snakes.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    continue
                if(snak.pos[1]==snak.boost[1] and snak.pos[0]==snak.boost[0]):
                    snak.pickupBoost()
                    snak.pickTimer=0
                    ge[x].fitness+=5
                    continue
                for we in snak.prevPoints:
                    if(we[0]==snak.pos[0] and we[1]==snak.pos[1]):
                        snak.dead=1
                        snak.size-=1
                        ge[x].fitness-=2
                        snakes.pop(x)
                        nets.pop(x)
                        ge.pop(x)
                    continue
                if(snak.pickTimer >=10000):
                    ge[x].fitness-=5
                    snakes.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    continue
                if(x<=len(ge)-1 and math.sqrt((snak.pos[0] - snak.boost[0])**2 + (snak.pos[1] - snak.boost[1])**2)/500 < 0.1):
                    ge[x].fitness+=distance
        if(len(snakes)==0): break
        text2 = font2.render('Generations: ' + str(snake.gens) + "   Alive: " + str(len(snakes)) + "   Fitness: " + str(max(fitnesses)), True, (255,0,0))
        window.blit(text2, (5,480))
        pygame.display.update()
        # Event manager
        for evt in pygame.event.get():    
            if evt.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def run(config):
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config)
    p=neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats=neat.StatisticsReporter()
    p.add_reporter(stats)
    winner=p.run(main,200)
    filename = "trainedData"
    print("Saving checkpoint to {0}".format(filename))
    with gzip.open(filename, 'w', compresslevel=5) as f:
        data = (winner)
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-feedforward.txt")
    run(config_path)
