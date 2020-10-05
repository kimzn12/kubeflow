#!/bin/bash -e
image_name=khw2126/quant-aware
image_tag=0.0.5
full_image_name=${image_name}:${image_tag}
base_image_tag=3.8-slim
cd "$(dirname "$0")"
docker build --build-arg BASE_IMAGE_TAG=$base_image_tag -t "$full_image_name" .
docker push "$full_image_name"
# Output the strict image name (which contains the sha256 image digest)
# This name can be used by the subsequent steps to refer to the exact image that was built even if another image with the same name was pushed
image_name_with_digest=$(docker inspect --format="{{index .RepoDigests 0}}" "$full_image_name")
strict_image_name_output_file=./versions/image_digests_for_tags/$image_tag
mkdir -p "$(dirname "$strict_image_name_output_file")"
echo $image_name_with_digest | tee "$strict_image_name_output_file"

