class MovementControler():
	def __call__(self, power, angles=None):
		self.power_back_left_engine, self.power_back_right_engine,
		self.power_front_left_engine, self.power_front_right_engine = power

		if angles == None:
			self.angle_left_servo = 90
			self.angle_middle_servo = 90
			self.angle_right_servo = 90
		else:
			self.angle_left_servo,
			self.angle_middle_servo,
			self.angle_right_servo = angles
		send_data()
	def send_data():
		data_to_send = [self.angle_middle_servo, self.angle_right_servo,
						self.power_front_left_engine, self.power_front_right_engine,
						self.power_back_left_engine, self.power_back_right_engine]
		bus.write_i2c_block_data(address, self.angle_left_servo, data_to_send)
	def return_state(self):
		return 
		np.array([[self.power_right_engine,self.power_left_engine],
		[self.angle_left_servo,self.angle_right_servo,self.angle_right_servo]])
	def __init__():
		pass 
if __name__=="__main__":
	bus=smbus.SMBus(1)
	SLAVE_ADDRESS = 0x20
	MC=MovementControler()

