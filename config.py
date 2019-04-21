config = dict()
config['actor_lr'] = 3e-3
config['critic_lr'] = 1e-3
config['actor_l2'] = 5e-3
config['critic_l2'] = 1e-3
config['batch_size'] = 64
config['betas'] = [0.9, 0.999]  # [beta1, beta2]

config['replay_size'] = 10 ** 5
config['history_len'] = 1
config['train_start'] = 10 ** 4
config['discount_factor'] = 0.99
config['tau'] = 1e-3
config['noise_size'] = 0.2

config['action_low'] = -1
config['action_high'] = 1

config['a_fc1'] = 32
config['a_fc2'] = 32
config['c_fc1'] = 32
config['c_fc2'] = 32

config['target_score'] = 2

config['env_path'] = 'Reacher/Reacher_single/Reacher.exe'
config['weights_save_path'] = 'save/weights/'
config['plot_save_path'] = 'save/reward.png'
config['rewards_save_path'] = 'save/reward.dmp'
config['print_every'] = 1
