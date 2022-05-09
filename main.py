from model import *

states = ('A','B')

#list of possible observations
possible_observation = ('1','2','3','4','5','6')

# The observations that we observe and feed to the model
observations = ('1','2','3','4')
obs4 = ('1', '3','5','6')

# observations
observation_tuple = []
observation_tuple.extend( [observations,obs4] )
quantities_observations = [10, 20]

start_probability = np.matrix( '0.5 0.5')
transition_probability = np.matrix('0.8 0.2 ;  0.3 0.7 ')
tmp_prob = 1/6
emission_probability = np.matrix( ' {} {} {} {} {} {}  ; 0.1 0.1 0.1 0.1 0.1 0.5  '.format(tmp_prob,tmp_prob,tmp_prob,tmp_prob,tmp_prob,tmp_prob) )
print(start_probability)
print(transition_probability)
print(emission_probability)
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
# print(e,t,s)
