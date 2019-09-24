from abc import ABC, abstractmethod


class RunnerBase(ABC):
    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def iteration_callback_(self, iteration, done, state, fitness, curve, user_data):
        # default implementation for testing
        print(f'iteration:[{iteration}], done:[{done}] - fitness:[{fitness:.8f}]')
        print(f'\tstate:[{state}]')
        print(f'\tuser_data:[{user_data}]')
        print()
