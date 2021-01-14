def robot_move_thread(e):
    with utilities.DeviceConnection.createTcpConnection(args) as router:
    	# Create required services
    	base = BaseClient(router)
    	base_cyclic = BaseCyclicClient(router)
        # Initial Movements
    	success = True
    	#Move into transport position (home)
    	home_pose=[184.19,291.7,171.7,213.3,181.8,45.8,266.6]
    	#success &= highlevel_movements.angular_action_movement(base,joint_positions=home_pose,speed=5)
    	success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[-0.06,-0.096,0.8,-90.823,171.8,113.73],speed=0.1)
    	#sys.exit()
    	# home_pose=[-0.154,0,0.34,-55.9,173.3,86.6]
    	# success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,waypoint=home_pose,speed=0.1)
    	#Exectue Sequence
    	#success &= highlevel_movements.call_kinova_web_sequence(base, base_cyclic,seq_name="grab_beer_v3")
    	#sys.exit()
    	#Execute Sequence V2
    	"""
    	speedj=5
    	speedl=0.1
    	success &= highlevel_movements.angular_action_movement(base,joint_positions=[89.695, 336.743, 176.642, 232.288, 180.629, 70.981, 272.165],speed=speedj)
    	success &= highlevel_movements.angular_action_movement(base,joint_positions=[91.583, 23.663, 174.547, 265.846, 180.949, 35.446, 272.106],speed=speedj)
    	success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.01796681061387062, -0.6095998287200928, 0.2607220709323883, -89.98168182373047, -176.41896057128906, 178.88327026367188],speed=speedl)
    	success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.015759950503706932, -0.6483935713768005, 0.2543827295303345, -91.34257507324219, 178.12986755371094, 178.2921905517578],speed=speedl)
    	# GRIPPER_ACTION
    	highlevel_movements.SendGripperCommands(base,value=0.7)
    	success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.016505524516105652, -0.65036940574646, 0.38649141788482666, -92.8912353515625, 178.2748565673828, 179.34559631347656],speed=speedl)
    	success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.07478067278862, -0.1246325895190239, 0.8614432215690613, 90.16726684570312, 9.310687065124512, 5.854083061218262],speed=speedl)
    	success &= highlevel_movements.angular_action_movement(base,joint_positions=[179.62, 307.879, 172.652, 289.174, 180.408, 69.243, 272.359],speed=speedj)

    	#move to surveillance position
    	success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[-0.06,-0.096,0.8,-90.823,171.8,113.73],speed=0.1)
    	"""
        time.sleep(1)
        e.set() #tell that the process is ready
        print("process is done")


def foo(bar, baz):
  print 'hello {0}'.format(bar)
  return 'foo' + baz

from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=1)

async_result = pool.apply_async(foo, ('world', 'foo')) # tuple of args for foo

# do some other stuff in the main process

return_val = async_result.get()  # get the return value from your function.


import Queue             # Python 2.x
#from queue import Queue # Python 3.x






















from threading import Thread

def foo(bar):
    print 'hello {0}'.format(bar)
    return 'foo'

que = Queue.Queue()      # Python 2.x
#que = Queue()           # Python 3.x

t = Thread(target=lambda q, arg1: q.put(foo(arg1)), args=(que, 'world!'))
t.start()
t.join()
result = que.get()
print result
