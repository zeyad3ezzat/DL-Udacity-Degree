import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        reward=np.tanh(1- 0.0005*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        return  reward
        '''reward = 1.-.001*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        if (abs(self.sim.pose[0] - self.target_pos[0])) < 0.25:
            reward += 0.03
        if (abs(self.sim.pose[1] - self.target_pos[1])) < 0.25:
            reward += 0.03
        if (abs(self.sim.pose[2] - self.target_pos[2])) < 0.25:
            reward += 0.03
        
        if self.sim.time < self.sim.runtime and self.sim.done == True:
            reward -= 10

        
        return reward'''
        """Uses current pose of sim to return reward."""

        
        
        ''' 
        x,y,z=self.sim.pose[:3]
        t_x,t_y,t_z=self.target_pos
        v_x,v_y,v_z=self.sim.v
        a_x,a_y,a_z=self.sim.angular_v
        penalty=(abs(v_x)+abs(v_y))*2
        reward+=(v_z*3)
        if v_x==0 or v_y==0:
            reward+=20
        if v_z!=0:
            reward+=10
        else:
            penalty*=2
        d=(abs(self.sim.pose[:3] - self.target_pos)).sum()
#         d_xy=(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()

        d_x=abs(t_x-x)
        d_y=abs(t_y-y)
        #it will be reward if it's below 10/100 penalty if greater 
        reward+=(10-d_x)    
        reward+=(10-d_y)  
#         reward+=(5-d_xy)    



            

        reward+=(30-d)
        return reward-penalty

#         reward = 1 - 0.0005*(abs(self.sim.pose[:3] - self.target_pos)).sum()
#         reward=0
#         penalty = 0
#         current_position = self.sim.pose[:3]
        # penalty for euler angles, we want the takeoff to be stable
#         penalty += abs(self.sim.pose[3:6]).sum()
        # penalty for distance from target
#         penalty += abs(current_position[0]-self.target_pos[0])**2
#         penalty += abs(current_position[1]-self.target_pos[1])**2
#         penalty += 10*abs(current_position[2]-self.target_pos[2])**2
#         reward = np.tanh(1 - 0.0005*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        # link velocity to residual distance
#         penalty += abs(abs(current_position-self.target_pos).sum() - abs(self.sim.v).sum())

#         distance = np.sqrt((current_position[0]-self.target_pos[0])**2 + (current_position[1]-self.target_pos[1])**2 + (current_position[2]-self.target_pos[2])**2)
        # extra reward for flying near the target
#         if distance < 10:
#             reward += 1000
        # constant reward for flying
#         reward += 100
#         reward = 0
#         factor = 3
#         dis = self.sim.pose[2] - self.target_pos[2]
        
#         if(dis >= 0):                 # agent above or equal the target
#             reward += dis * factor
#         else:                         # agent below the target   
#             reward += (1/np.abs(dis)) * factor
#         reward-=(1/(abs((self.sim.pose[0]-self.target_pos[0])+(self.sim.pose[0]-self.target_pos[0]))))
        # to make it in a range [-1,1]    
#         reward = np.tanh(reward)  

        #'''        
    '''
        reward = 0
        penalty = 0
        current_position = self.sim.pose[:3]
        # penalty for euler angles, we want the takeoff to be stable
        penalty += abs(self.sim.pose[3:6]).sum()
        penalty += abs(current_position[0]-self.target_pos[0])**3
        penalty += abs(current_position[1]-self.target_pos[1])**3
        penalty += abs(current_position[2]-self.target_pos[2])**2

        distance = np.sqrt((current_position[0]-self.target_pos[0])**2 + (current_position[1]-self.target_pos[1])**2 + (current_position[2]-self.target_pos[2])**2)
        if distance < 10 and distance>5:
            reward += 1000
        elif distance < 5 :
            reward += 2000


        reward += 100
        return reward - penalty*0.0002
'''

        


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += (self.get_reward()/2)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state