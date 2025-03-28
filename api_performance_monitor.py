import psutil
import matplotlib.pyplot as plt
import numpy as np
import time
import requests
import json
import os
from datetime import datetime

# Create directory for monitoring results
os.makedirs('monitoring', exist_ok=True)

def monitor_api_performance(num_requests=100, batch_size=10, endpoint="http://localhost:8000"):
    """
    Monitor API performance metrics including response time and memory usage
    
    Args:
        num_requests: Number of requests to send
        batch_size: Number of requests to send in each batch
        endpoint: API endpoint URL
    """
    print(f"Starting API performance monitoring with {num_requests} requests...")
    
    # Prepare test data
    spam_examples = [
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "PRIVATE! Your 2003 Account Statement for shows 800 un-redeemed S. I. M. points. Call 08719899230 Identifier Code: 41685 Expires 07/11/04",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info",
        "URGENT! Your Mobile No. was awarded £2000 Bonus Caller Prize on 5/9/03 This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU"
    ]
    
    ham_examples = [
        "Sounds good, Tom! I'll see you tomorrow at the meeting.",
        "Can you please pick up milk on your way home?",
        "The project deadline has been extended to next Friday.",
        "Happy birthday! Hope you have a great day!",
        "I'm running late for dinner, should be there in 15 minutes."
    ]
    
    # Lists to store metrics
    response_times = []
    memory_usage = []
    timestamps = []
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage.append(initial_memory)
    timestamps.append(0)
    
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Send requests and measure performance
    for i in range(num_requests):
        # Alternate between spam and ham examples
        message = spam_examples[i % len(spam_examples)] if i % 2 == 0 else ham_examples[i % len(ham_examples)]
        
        # Prepare request payload
        payload = {"message": message}
        
        # Record start time
        start_time = time.time()
        
        try:
            # Send request to API
            response = requests.post(f"{endpoint}/predict", json=payload)
            
            # Record response time
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Record memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory)
            timestamps.append(i + 1)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_requests} requests. Latest response time: {response_time:.4f}s")
            
            # Small delay to prevent overwhelming the API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error sending request {i+1}: {e}")
    
    # Calculate statistics
    avg_response_time = np.mean(response_times)
    min_response_time = np.min(response_times)
    max_response_time = np.max(response_times)
    p95_response_time = np.percentile(response_times, 95)
    
    min_memory = np.min(memory_usage)
    max_memory = np.max(memory_usage)
    memory_increase = max_memory - initial_memory
    
    # Print results
    print("\n===== API Performance Metrics =====")
    print(f"Total Requests: {num_requests}")
    print(f"Average Response Time: {avg_response_time:.4f} seconds")
    print(f"Minimum Response Time: {min_response_time:.4f} seconds")
    print(f"Maximum Response Time: {max_response_time:.4f} seconds")
    print(f"95th Percentile Response Time: {p95_response_time:.4f} seconds")
    print(f"Minimum Memory Usage: {min_memory:.2f} MB")
    print(f"Maximum Memory Usage: {max_memory:.2f} MB")
    print(f"Memory Usage Increase: {memory_increase:.2f} MB")
    
    # Create visualizations
    # 1. Response Time Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(response_times, bins=20, alpha=0.7, color='blue')
    plt.axvline(avg_response_time, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_response_time:.4f}s')
    plt.axvline(p95_response_time, color='green', linestyle='dashed', linewidth=2, label=f'95th Percentile: {p95_response_time:.4f}s')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('API Response Time Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('monitoring/response_time_distribution.png')
    
    # 2. Response Time Over Requests
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_requests + 1), response_times, marker='.', linestyle='-', alpha=0.7)
    plt.axhline(avg_response_time, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_response_time:.4f}s')
    plt.xlabel('Request Number')
    plt.ylabel('Response Time (seconds)')
    plt.title('API Response Time Over Requests')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('monitoring/response_time_trend.png')
    
    # 3. Memory Usage Over Requests
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, memory_usage, marker='.', linestyle='-', color='green', alpha=0.7)
    plt.xlabel('Request Number')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Requests')
    plt.grid(True, alpha=0.3)
    plt.savefig('monitoring/memory_usage.png')
    
    # Save metrics to file
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_requests": num_requests,
        "response_time": {
            "average": avg_response_time,
            "minimum": min_response_time,
            "maximum": max_response_time,
            "p95": p95_response_time
        },
        "memory_usage": {
            "initial": initial_memory,
            "minimum": min_memory,
            "maximum": max_memory,
            "increase": memory_increase
        }
    }
    
    with open('monitoring/performance_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nPerformance monitoring completed. Results saved to 'monitoring' directory.")
    
    return metrics

def simulate_daily_load(requests_per_day=10000, simulation_factor=0.01, endpoint="http://localhost:8000"):
    """
    Simulate daily load on the API and project performance metrics
    
    Args:
        requests_per_day: Expected number of requests per day
        simulation_factor: Fraction of daily requests to actually simulate
        endpoint: API endpoint URL
    """
    # Calculate number of requests to simulate
    num_requests = int(requests_per_day * simulation_factor)
    
    print(f"Simulating daily load of {requests_per_day} requests (actually running {num_requests} requests)...")
    
    # Run performance monitoring with the calculated number of requests
    metrics = monitor_api_performance(num_requests=num_requests, endpoint=endpoint)
    
    # Project daily metrics
    projected_metrics = {
        "requests_per_day": requests_per_day,
        "simulation_factor": simulation_factor,
        "simulated_requests": num_requests,
        "projected_daily_avg_response_time": metrics["response_time"]["average"],
        "projected_daily_p95_response_time": metrics["response_time"]["p95"],
        "projected_daily_memory_increase": metrics["memory_usage"]["increase"] * (requests_per_day / num_requests)
    }
    
    # Print projected metrics
    print("\n===== Projected Daily Performance =====")
    print(f"Expected Daily Requests: {requests_per_day}")
    print(f"Projected Average Response Time: {projected_metrics['projected_daily_avg_response_time']:.4f} seconds")
    print(f"Projected 95th Percentile Response Time: {projected_metrics['projected_daily_p95_response_time']:.4f} seconds")
    print(f"Projected Memory Increase: {projected_metrics['projected_daily_memory_increase']:.2f} MB")
    
    # Save projected metrics to file
    with open('monitoring/projected_daily_metrics.json', 'w') as f:
        json.dump(projected_metrics, f, indent=2)
    
    # Create dashboard visualization
    plt.figure(figsize=(15, 10))
    
    # Response Time subplot
    plt.subplot(2, 2, 1)
    plt.bar(['Average', 'P95'], 
            [projected_metrics['projected_daily_avg_response_time'], 
             projected_metrics['projected_daily_p95_response_time']], 
            color=['blue', 'orange'])
    plt.ylabel('Response Time (seconds)')
    plt.title('Projected Response Times')
    plt.grid(True, alpha=0.3)
    
    # Memory Usage subplot
    plt.subplot(2, 2, 2)
    plt.bar(['Initial', 'Projected Increase'], 
            [metrics['memory_usage']['initial'], 
             projected_metrics['projected_daily_memory_increase']], 
            color=['green', 'red'])
    plt.ylabel('Memory (MB)')
    plt.title('Projected Memory Usage')
    plt.grid(True, alpha=0.3)
    
    # Requests per second subplot
    plt.subplot(2, 2, 3)
    requests_per_second = requests_per_day / (24 * 60 * 60)
    plt.bar(['Requests per Second'], [requests_per_second], color='purple')
    plt.ylabel('Requests')
    plt.title('Projected Request Rate')
    plt.grid(True, alpha=0.3)
    
    # Server capacity subplot
    plt.subplot(2, 2, 4)
    max_requests_per_second = 1 / metrics['response_time']['average']
    capacity_percentage = (requests_per_second / max_requests_per_second) * 100
    plt.pie([capacity_percentage, 100 - capacity_percentage], 
            labels=['Used', 'Available'], 
            colors=['red', 'lightgray'],
            autopct='%1.1f%%',
            startangle=90)
    plt.axis('equal')
    plt.title('Server Capacity Utilization')
    
    plt.tight_layout()
    plt.savefig('monitoring/daily_performance_dashboard.png')
    
    print("\nDaily load simulation completed. Results saved to 'monitoring' directory.")
    
    return projected_metrics

if __name__ == "__main__":
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("API is running. Starting performance monitoring...")
            
            # Run performance monitoring
            metrics = monitor_api_performance(num_requests=50)
            
            # Simulate daily load
            projected_metrics = simulate_daily_load(requests_per_day=10000, simulation_factor=0.005)
        else:
            print(f"API returned status code {response.status_code}. Please make sure the API is running correctly.")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Please make sure the API is running on http://localhost:8000 before running this script.")
