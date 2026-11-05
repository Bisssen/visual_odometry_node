#!/usr/bin/env python3
import os
import subprocess
import signal
import time

# --- Config ---
BAGS_DIR = os.path.expanduser('~/ros/scout_ws/rosbags')
OUTPUT_DIR = os.path.expanduser('~/ros/scout_ws/src/visual_odometry_node/scripts')
COUNTER_FILE = os.path.expanduser('~/.vo_test_counter')
TIMEOUT = 30
SKIP_BAGS = {'voxlbag_encoded_20260320_080401', 'rosbag2_2026_09_10-09_55_36'}


def get_next_index():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r') as f:
            idx = int(f.read().strip())
    else:
        idx = 0
    idx += 1
    with open(COUNTER_FILE, 'w') as f:
        f.write(str(idx))
    return idx


def kill_existing_vo():
    subprocess.run(['pkill', '-f', 'visual_odometry_node'], check=False)
    time.sleep(1)

def run_bag(bag_path):
    kill_existing_vo()

    idx = get_next_index()
    bag_name = os.path.basename(bag_path)
    csv_name = os.path.join(OUTPUT_DIR, f'trajectory_{idx}_{bag_name}.csv')
    print(f'\n--- Run #{idx}: {bag_name} ---')
    print(f'Saving to {csv_name}')

    procs = []

    try:
        # 1. VO node
        procs.append(subprocess.Popen(
            ['ros2', 'run', 'visual_odometry_node', 'visual_odometry_node'],
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
        ))

        time.sleep(2)

        # 2. Trajectory recorder
        procs.append(subprocess.Popen([
            'python3',
            os.path.join(os.path.dirname(__file__), 'plot.py'),
            csv_name
        ]))

        time.sleep(1)

        # 3. Bag replay
        bag_proc = subprocess.Popen(
            ['ros2', 'bag', 'play', bag_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        procs.append(bag_proc)

        # try:
        #     bag_proc.wait(timeout=10)
        # except subprocess.TimeoutExpired:
        #     bag_proc.send_signal(signal.SIGINT)
        #     bag_proc.wait()
        #     print('Bag stopped after 10s')

        bag_proc.wait()
        # _, stderr = bag_proc.communicate()
        # print(f'Bag returncode: {bag_proc.returncode}')
        # if stderr:
        #     print(f'Bag stderr: {stderr.decode()}')

        print(f'Bag finished, waiting {TIMEOUT}s for VO to flush...')
        time.sleep(TIMEOUT)

    except KeyboardInterrupt:
        print(f'\nInterrupted during {bag_name}')
        raise

    finally:
        for p in procs:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print(f'Saved: {csv_name}')


def main():
    bags = sorted([
        os.path.join(BAGS_DIR, d)
        for d in os.listdir(BAGS_DIR)
        if os.path.isdir(os.path.join(BAGS_DIR, d))
        and d not in SKIP_BAGS
    ])

    print(f'Found {len(bags)} bags:')
    for b in bags:
        print(f'  {os.path.basename(b)}')

    for bag in bags:
        try:
            run_bag(bag)
        except KeyboardInterrupt:
            print('\nAborted by user.')
            break
        except Exception as e:
            print(f'Error on {bag}: {e} — skipping')
            continue

    print('\nAll done.')


if __name__ == '__main__':
    main()