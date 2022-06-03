
docker build --network host $(env | cut -f1 -d= | grep -E '_(proxy|REPO|VER)$' | sed 's/^/--build-arg /') --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t preprocess:1.0 -f Dockerfile .
