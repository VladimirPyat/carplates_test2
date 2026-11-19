
import argparse

from dataset_prepare import stage0_hash_dedup

DEFAULT_OUT = "dataset_prepare/audit_out"  # корень артефактов; этапы кладут в подпапки по имени модуля

def main():
    """
    CLI подготовки датасета. Только разбор команд, логика — в пакете dataset_prepare
    Новые этапы если надо добавляем <filename>.register(sub) """
    ap = argparse.ArgumentParser(prog="dataset_prepare", description="Подготовка датасета по этапам")
    sub = ap.add_subparsers(dest="command", required=True)
    stage0_hash_dedup.register(sub)
    stage1_audit_files.register(sub)
    args = ap.parse_args()
    args.out = args.out or DEFAULT_OUT
    args.func(args)


if __name__ == "__main__":
    main()