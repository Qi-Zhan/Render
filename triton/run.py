import subprocess
import os
import time

cases = [
    "1924",
    "5990",
    "3017",
    "5895",
    "4551",
    "1190",
    "2843",
    "1960",
    "2680b",
    "376",
    "5065",
    "1840",
    "1666",
    "1937",
    "1671",
    "1821",
    "4701",
    "1808",
    "1578",
    "6227",
]

total_start = time.time()
times = []

for case in cases:
    print(f"\n=== Running {case} ===")
    start = time.time()

    env = os.environ.copy()
    env["TRITON_INTERPRET"] = "1"

    command = ["python3", f"{case}_interval.py"]

    try:
        result = subprocess.run(
            command,
            cwd=f"./{case}",
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        elapsed = time.time() - start
        if result.stderr:
            print(f"[stderr]\n{result.stderr}")
        print(f"✅ Finished {case} in {elapsed:.2f} seconds")
        times.append(elapsed)

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"❌ Error occurred while running {case} (in {elapsed:.2f} seconds):")
        print(f"[stderr]\n{e.stderr}")

total_elapsed = time.time() - total_start
print(f"\n=== All cases finished in {total_elapsed:.2f} seconds ===")

print("each time:")
print(times)
