import pickle
import torch
from typing import Dict, List, Tuple

class CPU_Unpickler(pickle.Unpickler):
    """Fallback PyTorch unpickler for loading GPU pickles on a CPU machine."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import io
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class AirIMULoader:
    """
    A helper class to load AirIMU .pkl output files and format them temporally 
    for the real-time IMUContext.step() method.
    """
    def __init__(self, pkl_path: str):
        """
        Loads the AirIMU pickle file which contains the networks corrected IMU readings 
        and output covariances from the offline preprocessing step.
        """
        with open(pkl_path, 'rb') as f:
            try:
                self.data = pickle.load(f)
            except Exception:
                f.seek(0)
                self.data = CPU_Unpickler(f).load()

    def get_sequence(self, sequence_name: str) -> Dict:
        """Returns the AirIMU data block for a specific dataset sequence."""
        if sequence_name not in self.data:
            raise KeyError(f"Sequence '{sequence_name}' not found in AirIMU pickle.")
        return self.data[sequence_name]

    def extract_camera_window(self, 
                              airimu_seq_data: Dict, 
                              raw_seq_data: Dict, 
                              start_idx: int, 
                              end_idx: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Extracts a chunk of temporally aligned IMU data between two camera frames and formats
        it correctly for IMUContext.step().
        
        Args:
            airimu_seq_data: The specific dataset sequence loaded from the AirIMU pkl via get_sequence()
            raw_seq_data: The EXACT same dataset sequence block from your RAW hardware IMU dataloader
            start_idx: The starting index in the IMU array (corresponding to camera frame t-1)
            end_idx: The ending index in the IMU array (corresponding to camera frame t)
            
        Returns:
            corrected_imu: list of dicts formatted for the EKF pipeline 
            raw_imu: list of dicts formatted for the Air-IO deep learning pipeline
        """
        corrected_imu = []
        raw_imu = []
        
        for i in range(start_idx, end_idx):
            # 1. Format the Corrected Tick for the EKF Simulation
            # Fallback to the raw readings if your specific AirIMU pickle doesn't explicitly rewrite them.
            corr_tick = {
                "acc": airimu_seq_data["corrected_acc"][i] if "corrected_acc" in airimu_seq_data else raw_seq_data["acc"][i],
                "gyro": airimu_seq_data["corrected_gyro"][i] if "corrected_gyro" in airimu_seq_data else raw_seq_data["gyro"][i],
                
                # Load Network Covariances if they exist
                "acc_cov": airimu_seq_data["acc_cov"][i] if "acc_cov" in airimu_seq_data else None,
                "gyro_cov": airimu_seq_data["gyro_cov"][i] if "gyro_cov" in airimu_seq_data else None,
                
                # Fetch explicit timestamp deltas
                "dt": raw_seq_data["dt"][i] if "dt" in raw_seq_data else 0.005
            }
            corrected_imu.append(corr_tick)
            
            # 2. Format the Raw Tick for the deep-learning Air-IO feature extractor
            raw_tick = {
                "acc": raw_seq_data["acc"][i],
                "gyro": raw_seq_data["gyro"][i],
            }
            raw_imu.append(raw_tick)
            
        return corrected_imu, raw_imu
