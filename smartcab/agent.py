import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = {}
        self.epsilon =.2
        self.epsilon_annealing_rate = .01
        self.episilon_reset_trials = 200
        self.alpha = 0.6
        self.gama = 0.4
        self.q_table = {}
        self.valid_actions = self.env.valid_actions
        self.total_wins = 0
        self.trial_infractions = 0
        self.infractions_record = []
        self.trial_count = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.infractions_record.append(self.trial_infractions)
        self.trial_infractions = 0
        self.trial_count += 1
        if self.trial_count < self.episilon_reset_trials:
            self.epsilon = .05

    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.set_current_state(inputs)

        # TODO: Select action according to your policy
        action = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def performace_report(self, n_trials):

        print '\n'+ 25*'*' + "FINAL REPORT:" + 25*'*'
        print 'AMOUNT OF TIMES REACHED GOAL:', self.total_wins
        print 'TRAFFIC INFRACTIONS RECORD:', self.infractions_record
        print 'AMOUNT OF TIMES GOAL NOT REACHED', (n_trials - self.total_wins)
        print 'TOTAL AMOUNT OF TRAFFIC INFRACTIONS:', sum(self.infractions_record)
        print 25*'*'+'\n'

    def save_states(self,reward):
        if reward >= 5:
            self.total_wins += 1
        if reward <= 1:
            self.trial_infractions += 1
        else:
            pass

    def set_current_state(self, inputs):
        return {"light": inputs["light"], "oncoming": inputs["oncoming"], "left": inputs["left"], "next_waypoint": self.next_waypoint
        }

    def choose_action(self, state):
        if random.random() < self.epsilon:
            self.epsilon -= self.epsilon_annealing_rate
            return random.choice(self.valid_actions)

        opt_action = self.valid_actions[0]
        opt_value = 0

        for action in self.valid_actions:
            cur_value = self.q_value(state, action)
            if cur_value > opt_value:
                opt_action = action
                opt_value = cur_value
            elif cur_value == opt_value:
                opt_action = random.choice([opt_action, action])
        return opt_action

    def max_q_value(self, state):
        max_value = None
        for action in self.valid_actions:
            cur_value = self.q_value(state, action)
            if max_value is None or cur_value > max_value:
                max_value = cur_value
        return max_value

    def q_value(self, state, action):
        q_key = self.q_key(state, action)
        if q_key in self.q_table:
            return self.q_table[q_key]
        return 0

    def update_q_table(self, state, action, reward):
        """Updates q_table from action taken and reward recieved"""

        q_key = self.q_key(state, action)
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        new_state = self.set_current_state(inputs)

        #update q_value in q_table according to learning formula
        x = self.q_value(state, action)
        V = reward + (self.gamma * self.max_q_value(new_state))
        new_q_value = x + (self.alpha * (V - x))
        self.q_table[q_key] = new_q_value

    def q_key(self, state, action):
        return "{}.{}.{}.{}.{}".format(state["light"], state["next_waypoint"], state["oncoming"], state["left"], action)

def run(num_trials):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  
    # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=num_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    a.performace_report(num_trials)

if __name__ == '__main__':
    run(100)
