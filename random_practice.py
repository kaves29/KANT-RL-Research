import json

class Experiment:
    experiment_counter = 0
    def __init__(self, name, environment, rewards):
        self.name = name
        self.environment = environment
        self.rewards = rewards
        Experiment.experiment_counter += 1

    def average_reward(self):
        total = 0
        counter = 0
        for r in self.rewards:
            total += r
            counter += 1
        return total / counter
    
    def summary(self):
        print(f"Name: {self.name}, Environment: {self.environment}, Rewards: {self.rewards}, Average Reward: {self.average_reward()}")

    def save(self):
        data = {
            "name": self.name,
            "environment": self.environment,
            "rewards": self.rewards
        }
        with open(f"{self.name}.json", "w") as f:
            json.dump(data, f)

    @classmethod
    def get_experiment_count(cls):
        return f"Experiments: {cls.experiment_counter}"
    
class RLExperiment(Experiment):
    def __init__(self, name, environment, rewards, algorithm):
        self.algorithm = algorithm
        super().__init__(name, environment, rewards)

    def summary(self): # forgot how to do this
        super().summary()
        print(f"Algorithm: {self.algorithm}")
    
    def save(self): # forgot how to do this
        data = {
            "name": self.name,
            "environment": self.environment,
            "rewards": self.rewards,
            "algorithm": self.algorithm
        }
        with open(f"{self.name}.json", "w") as f:
            json.dump(data, f)

def load_experiment(filename):
    try:
        with open(filename) as f:
            data = json.load(f)
            print(data)
    except FileNotFoundError: 
        print("The file was not found! ")
    
experiment1 = Experiment("RL", "Moon", [1,8,15])
experiment2 = Experiment("KAN", "Circus", [6, 7, 8])
rlexperiment1 = RLExperiment("MLP", "School", [3, 8, 3], "basic")
rlexperiment2 = RLExperiment("KAN", "Claude HQ", [20, 22, 15], "advanced")

experiment1.summary()
experiment2.summary()
rlexperiment1.summary()
rlexperiment2.summary()

experiment1.save()
experiment2.save()
rlexperiment1.save()
rlexperiment2.save()

load_experiment("MLP.json")
load_experiment("KAN.json")

try:
    load_experiment("RL.json")
except:
    print("File does not exist/not found")
    
    
