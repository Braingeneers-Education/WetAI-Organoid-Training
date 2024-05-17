
import numpy as np
import csv
import time

# Fake class to test the method
class OrganoidInterface:
    def __init__(self, game_env, verbose=False, test_file_name='test', n_episodes=10):

        class DummyEnv:
            def __init__(self):
                self.save_file = None
                self.reset = lambda : None
                # self.step = lambda a, b,c : return None, None, None, None
                # step function takes in 2 arguments, action and tag, and returns obs, done
                self.step = lambda action=None, tag=None : (None, None)

        self.game_env = game_env
        # Env is dummy for now
        self.env = DummyEnv() # Dummy maxwell interface environment

        self.save_file = test_file_name
        self.sensory_neurons = np.arange(3,10) # Max 10 sensory neurons to stimulate
        self.sensory_stim_Hz = np.zeros(len(self.sensory_neurons)) # Rate of stimulation

        self.training_neurons = np.arange(5,12) # Max 10 training neurons to stimulate

        self.motor_neurons = np.arange(3)# Indices of motor neurons
        self.motor_spike_count = np.zeros(len(self.motor_neurons)) # Stores the spike count for each motor neuron
        self.motor_spike_rate = np.zeros(len(self.motor_neurons)) # Stores the spike rate for each motor neuron

        self.verbose = verbose


        self.fake_spikes = np.random.uniform(0, 1, (1000,3))
        self.fake_spikes[self.fake_spikes < 0.9] = 0
        self.fake_spikes[self.fake_spikes > 0] = 1
        self.state = 'read'

        self.read_period_ms = 200 # Time between simulation steps
        self.train_period_ms = 200 # Time to give training signal
        self.wait_period_ms = 1000 # Time to wait after training

        self.rewards = []
        self.episode = 0
        self.n_episodes = n_episodes
        self.max_time = 1000

        # Set connectivity matrix
        self.connectivity = None
        self.connectivity = self.get_dummy_causal_connectivity()

        self.neuron_labels = []
        for i in range(self.N):
            cur_label = ''
            if i in self.sensory_neurons:
                cur_label += 'S'
            if i in self.training_neurons:
                cur_label += 'T'
            if i in self.motor_neurons:
                cur_label += 'M'
            self.neuron_labels.append(cur_label)
            

        

    def time_elapsed(self):
        return time.perf_counter() - self.start_time

    def run(self):
        done = False
        just_entered = True
        total_reward = 0
        do_training = True

        # Loggers
        # ---- Game log ----
        game_log = open(self.save_file + '_game_log.csv','w')
        # Header for csv
        game_logger = csv.writer(game_log)
        game_logger.writerow(['time','food_signal', 'spike_signal', 'agent_pos_x', 'agent_pos_y',
                                'agent_dir', 'food_got', 'spike_hit', 'reward',
                                              'episode', 'moving_speed', 'turning_speed'])
    
        # ---- Training log ----
        train_log = open(self.save_file + '_train_log.csv','w')
        # Header for csv
        train_logger = csv.writer(train_log)
        train_logger.writerow(['time','pattern','reward', 'episode'])


        self.start_time = time.perf_counter()
       
        # ================== Main Loop ==================
        while not done:
            # ~~~~~~~~~~~~~~~~~~~ Read phase Logic ~~~~~~~~~~~~~~~~~~~
            if self.state == 'read':
                # ================== Enter read phase ==================
                if just_entered:
                    just_entered = False
                    
                # ================== Sensory stimulation logic ==================
                if self.verbose:
                    print("Stimming sensory neurons with", self.sensory_stim_Hz)

                # ================== Spike Read Logic ==================
                self.motor_spike_count = self.get_fake_motor_spikes()

             # ================== Exit read phase ==================
             # This would happen after the read period is up
                if self.verbose:
                    print('Spike count: ', self.motor_spike_count, f'Rate {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f}')
                self.state = 'game'
                just_entered = True
                continue
                    
            # ~~~~~~~~~~~~~~~~~~~ Game phase Logic ~~~~~~~~~~~~~~~~~~~
            elif self.state == 'game':
                if just_entered:
                    game_action = self.get_motor_signal()
                    self.game_obs,reward,game_done,inf = self.game_env.step(game_action)
                    
                    total_reward += reward


                    # Converting the game observation to sensory stimulation
                    self.set_sensory_signal(self.game_obs)
                    self.cause_fake_motor_spikes()

                    # Log the game observation
                    game_logger.writerow([time.time(), *self.game_obs, reward, self.episode, game_action[0], game_action[1]])

                    # If game is done, go to train phase, else continue reading
                    if game_done:
                        self.state = 'train'
                        just_entered = True
                        continue
                    else:
                        self.state = 'read'
                        just_entered = True
                        continue

            # ~~~~~~~~~~~~~~~~~~~ Train phase Logic ~~~~~~~~~~~~~~~~~~~
            elif self.state == 'train':
                # ================== Enter train phase ==================
                if just_entered:
                    just_entered = False
                    self.rewards.append(total_reward)
                    do_training = True # Train 100% of the time
                   

                    self.episode += 1
                    if self.episode >= self.n_episodes:
                        done = True
                        return

                    if self.verbose:
                        print('Reward:', total_reward)
                        print('All rewards:', self.rewards)
                        print('-'*20)                        
                    # ================== Set Training pulses ==================
                    if do_training:
                        # Get pattern/update
                        train_pulse, train_freq = self.get_training_signal()
                        # This modifies the connectivity
                        # self.update_connectivity(train_pulse)

                        obs, done = self.env.step(action=train_pulse, tag='train') 
                        # Log the pattern
                        train_logger.writerow([self.time_elapsed(), train_pulse, total_reward, self.episode])
                        continue
                       
                # ================== Training logic ==================
                # There would be logic on stimming the training pulse at the proper frequency
                if self.verbose:
                    print("Stimming training neurons with", train_freq)
                    print("Training pulse:", train_pulse)
                    print("\tfor", self.train_period_ms, "ms")

                
                # ================== Exit train phase ==================
                # This would happen after the train period is up
                self.game_obs = self.game_env.reset()
                self.set_sensory_signal(self.game_obs)
                total_reward = 0
                self.state = 'wait'
                just_entered = True
                continue

                

            # ~~~~~~~~~~~~~~~~~~~ Wait phase Logic ~~~~~~~~~~~~~~~~~~~
            elif self.state == 'wait':
                # There would be logic on waiting for the proper time
                obs, done = self.env.step()
                if self.verbose:
                    print("Waiting for", self.wait_period_ms, "ms")
                if self.verbose:
                    print("Episode done")
                self.state = 'read'
                just_entered = True
                continue


        # Close the loggers
        game_log.close()
        train_log.close()
        

        return True
    
    def get_fake_motor_spikes(self):
        
        # use connectiviy matrix to get the spikes from sensory_stim_Hz
        
        # sensory_stim_Hz is a vector of shape (len(sensory_neurons),)
        # motor spike count is of shape (len(motor_neurons),)
        # connectivity is of shape (N, N), where 
        sub_connectivity = self.connectivity[self.sensory_neurons][:,self.motor_neurons]
        # mat mult plus random
        motor_spikes = np.dot(self.sensory_stim_Hz, sub_connectivity)*2

        # 1/4 power
        motor_spikes = motor_spikes ** 0.4

        # cast to int
        motor_spikes = motor_spikes.astype(int)
        # add some noise
        motor_spikes += np.random.randint(-2, 3, len(self.motor_neurons))
        # clip to 0
        motor_spikes = np.clip(motor_spikes, 0, None)
        self.motor_spike_rate = motor_spikes
        return motor_spikes
    
    def cause_fake_motor_spikes(self):
        self.motor_spike_count = self.get_fake_motor_spikes()
        self.motor_spike_rate = self.motor_spike_count / 1000


    def get_dummy_causal_connectivity(self):
        if self.connectivity is not None:
            return self.connectivity
        n_strong = 10
        # Get unique neurons in sensory and training, since there can be overlap
        unique_neurons = np.unique(np.concatenate([self.sensory_neurons, self.training_neurons]))
        self.N = len(unique_neurons) + len(self.motor_neurons)
        # Create a random connectivity matrix, most are 0-.3, some are >.6
        connectivity = np.random.uniform(0, .1, (self.N, self.N))
        # connectivity = np.zeros((self.N, self.N))
        # pick 5 random indexes and make them >.6
        connectivity[np.random.randint(0, self.N, n_strong), np.random.randint(0, self.N, n_strong)] = np.random.uniform(.6, 1, n_strong)
        self.connectivity = connectivity
        return connectivity



    def set_sensory_signal(self, obs):
        raise NotImplementedError
    
    def get_motor_signal(self):
        raise NotImplementedError
    
    def get_training_signal(self):
        raise NotImplementedError

    def set_sensory_function(self, func):
        '''
        Set the function to use to map the game observation to the sensory neurons
        '''
        self.set_sensory_signal = lambda game_env_obs: func(self, game_env_obs)

    def set_motor_function(self, func):
        '''
        Set the function to use to map the motor neurons to the action
        '''
        self.get_motor_signal = lambda : func(self)

    def set_training_function(self, func):
        '''
        Set the function to use to map the training neurons to the action
        '''
        self.get_training_signal = lambda : func(self)