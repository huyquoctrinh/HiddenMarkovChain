from model import *

states = ('s', 't')

#list of possible observations
possible_observation = ('A','B' )

# The observations that we observe and feed to the model
observations = ('A', 'B','B','A')
obs4 = ('B', 'A','B')

# observations
observation_tuple = []
observation_tuple.extend( [observations,obs4] )
quantities_observations = [10, 20]

start_probability = np.matrix( '0.5 0.5')
transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )
test = HMM(states,possible_observation,start_probability,transition_probability,emission_probability)
forw_prob = (test.forward(observations))
forw_prob = round(forw_prob, 5)
print(forw_prob)
vit_out = (test.viterbi(observations))
print(vit_out)
prob = test.log_prob(observation_tuple, quantities_observations)
prob = round(prob, 3)
print(prob)
num_iter=1000
e,t,s = test.BW(observation_tuple,num_iter,quantities_observations)