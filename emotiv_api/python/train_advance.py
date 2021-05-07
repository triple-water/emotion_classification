from cortex import Cortex

class TrainAdvance():
	def __init__(self):
		self.c = Cortex(user, debug_mode=True)
		self.c.do_prepare_steps()

	def get_active_action(self, profile_name):
		self.c.get_mental_command_active_action(profile_name)

	def get_command_brain_map(self, profile_name):
		self.c.get_mental_command_brain_map(profile_name)

	def get_training_threshold(self):
		self.c.get_mental_command_training_threshold(profile_name)

# ---------------------------------------------------

user = {
	"license" : "30a7df02-37ba-462c-8b75-feedfee8c815",
	"client_id" : "mm6xyIddErBCPW8lZFlAlG5k0f771wTzuvDT37fC",
	"client_secret" : "AmGvdjxk17EgLOwYbmJosCch0JfXevUNDSc3LPne0gKbtKFbO89K5x5nphIQd6JJ5yewvHiXhhSdoHUZd7uceXZeHsdXdoZQlWUsMqEnUltome8bVKScgBRNZEQOpZ6D",
	"debit" : 100
}

t = TrainAdvance()

profile_name = r'C:\Users\A\Desktop\test'

t.get_active_action(profile_name)
t.get_command_brain_map(profile_name)
t.get_training_threshold()

# ---------------------------------------------------