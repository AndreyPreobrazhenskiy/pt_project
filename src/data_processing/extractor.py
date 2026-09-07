import gzip
import csv
from collections import defaultdict
import polars as pl
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract():
    data_path = Path("flows.txt.gz")
    output_path = Path("data/host_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: {
        'dst_hosts': set(),
        'dst_ports': set(),
        'total_bytes': 0,
        'total_packets': 0,
        'total_duration': 0.0,
        'tcp_count': 0,
        'total_flows': 0
    })

    # ограничение 3 млн строк для баланса скорости и репрезентативности
    max_rows = 3_000_000 
    rows_processed = 0

    try:
        with gzip.open(data_path, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if rows_processed >= max_rows:
                    break
                
                if len(row) < 9:
                    continue
                    
                src = row[2]
                dst = row[4]
                dst_port = row[5]
                
                try:
                    protocol = int(row[6])
                    packets = int(row[7])
                    bytes_val = int(row[8])
                    duration = float(row[1]) if row[1] else 0.0
                except ValueError:
                    continue

                stats[src]['dst_hosts'].add(dst)
                stats[src]['dst_ports'].add(dst_port)
                stats[src]['total_bytes'] += bytes_val
                stats[src]['total_packets'] += packets
                stats[src]['total_duration'] += duration
                if protocol == 6:
                    stats[src]['tcp_count'] += 1
                stats[src]['total_flows'] += 1
                
                rows_processed += 1
                if rows_processed % 500_000 == 0:
                    logging.info(f"Обработано строк: {rows_processed:,}")

        logging.info(f"Чтение завершено. Всего строк: {rows_processed:,}")
        logging.info(f"Уникальных source хостов собрано: {len(stats)}")
        logging.info("Преобразование в DataFrame и сохранение...")

        data_list = []
        for src, s in stats.items():
            tcp_ratio = s['tcp_count'] / s['total_flows'] if s['total_flows'] > 0 else 0.0
            avg_duration = s['total_duration'] / s['total_flows'] if s['total_flows'] > 0 else 0.0
            
            data_list.append({
                'src_computer': src,
                'unique_dst_hosts': len(s['dst_hosts']),
                'unique_dst_ports': len(s['dst_ports']),
                'total_bytes': s['total_bytes'],
                'total_packets': s['total_packets'],
                'avg_duration': avg_duration,
                'tcp_ratio': tcp_ratio
            })

        df = pl.DataFrame(data_list)
        df.write_parquet(output_path)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logging.info(f"Данные успешно сохранены в {output_path}")
        logging.info(f"Размер итогового файла: {size_mb:.2f} МБ")
        print("\nПризнаки хостов (первые 10):")
        print(df.head(10))

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract()
