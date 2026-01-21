resource "aws_s3_bucket" "app_data" {
  bucket = "company-internal-assets-2024"
  tags = {
    Environment = "Dev"
  }
}