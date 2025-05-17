from fastmcp import FastMCP
import synexa
import os
import httpx
import asyncio
import openai
import boto3
import io
import uuid
import json
import base64

mcp = FastMCP("FileAndImageServer")

@mcp.tool()
async def generate_3d_model(
    prompt: str,
    output_filename: str = "generated_model.glb",
) -> str:
    """
    Generates a 3D model from a text prompt.
    Stage 1: Generates front, back, and side view images using OpenAI DALL-E.
    Stage 2: Uploads these images to a Tigris S3-compatible store.
    (Stages 3-6 for actual 3D model generation and download will be implemented next)
    """
    # --- Stage 1: Generate images with OpenAI gpt-image-1 ---
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return json.dumps({"error": "OPENAI_API_KEY environment variable not set."})

    try:
        aclient = openai.AsyncOpenAI(api_key=openai_api_key)
        image_data_for_s3 = {}
        base_prompt_for_views = f"{prompt}, highly detailed on a transparent background"
        
        # Generate front view first
        front_prompt = f"{base_prompt_for_views}, front view"
        print(f"Generating front view with OpenAI gpt-image-1: {front_prompt}")
        response_front = await aclient.images.generate(
            model="gpt-image-1",
            prompt=front_prompt,
            n=1,
            size="1024x1024",
            background="transparent",
            quality="high"
        )
        print(f"OpenAI response for front view: {response_front}")

        if not response_front.data or not response_front.data[0].b64_json:
            error_message = "Failed to get b64_json data from OpenAI for front view."
            print(f"ERROR: {error_message} OpenAI full response: {response_front}")
            return json.dumps({"error": error_message, "openai_response": str(response_front)})
        
        front_b64_json = response_front.data[0].b64_json
        front_image_bytes = base64.b64decode(front_b64_json)
        image_data_for_s3["front"] = io.BytesIO(front_image_bytes)
        print("Successfully decoded base64 image data for front view.")

        view_prompts_remaining = {
            "back": f"{base_prompt_for_views}, back view of the same subject",
            "side": f"{base_prompt_for_views}, side view of the same subject (profile view)"
        }

        for view, view_prompt in view_prompts_remaining.items():
            print(f"Generating {view} view with OpenAI gpt-image-1: {view_prompt}")
            response = await aclient.images.generate(
                model="gpt-image-1",
                prompt=view_prompt,
                n=1,
                size="1024x1024",
                background="transparent",
                quality="high"
            )
            print(f"OpenAI response for {view} view: {response}")

            if not response.data or not response.data[0].b64_json:
                error_message = f"Failed to get b64_json data from OpenAI for {view} view."
                print(f"ERROR: {error_message} OpenAI full response: {response}")
                return json.dumps({"error": error_message, "openai_response": str(response)})
            
            view_b64_json = response.data[0].b64_json
            view_image_bytes = base64.b64decode(view_b64_json)
            image_data_for_s3[view] = io.BytesIO(view_image_bytes)
            print(f"Successfully decoded base64 image data for {view} view.")

    except openai.APIError as e:
        return json.dumps({"error": f"OpenAI API error: {e}"})
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP error downloading image from OpenAI: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Error during OpenAI image generation/download: {type(e).__name__} - {e}"})

    # --- Stage 2: Upload images to Tigris or S3 ---
    tigris_endpoint_url = os.environ.get("TIGRIS_ENDPOINT_URL")
    tigris_access_key_id = os.environ.get("TIGRIS_ACCESS_KEY_ID")
    tigris_secret_access_key = os.environ.get("TIGRIS_SECRET_ACCESS_KEY")
    tigris_bucket_name = os.environ.get("TIGRIS_BUCKET_NAME")

    if not all([tigris_endpoint_url, tigris_access_key_id, tigris_secret_access_key, tigris_bucket_name]):
        return json.dumps({"error": "Tigris S3 environment variables not fully set (TIGRIS_ENDPOINT_URL, TIGRIS_ACCESS_KEY_ID, TIGRIS_SECRET_ACCESS_KEY, TIGRIS_BUCKET_NAME)."})

    s3_uploaded_urls = {}
    try:
        def upload_to_s3(view_name, image_data_io):
            s3_client = boto3.client(
                's3',
                endpoint_url=f"https://{tigris_endpoint_url}",
                aws_access_key_id=tigris_access_key_id,
                aws_secret_access_key=tigris_secret_access_key,
                region_name='auto'
            )
            object_key = f"generated_views/{prompt[:20].replace(' ','_')}_{view_name}_{uuid.uuid4()}.png"
            image_data_io.seek(0)
            s3_client.upload_fileobj(
                image_data_io,
                tigris_bucket_name,
                object_key,
                ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/png'}
            )
            public_url = f"https://{tigris_bucket_name}.{tigris_endpoint_url}/{object_key}"
            print(f"Successfully uploaded {view_name} image to Tigris: {public_url}")
            return public_url

        for view, img_io_data in image_data_for_s3.items():
            s3_uploaded_urls[f"{view}_image_url"] = await asyncio.to_thread(
                upload_to_s3, view, img_io_data
            )

    except Exception as e:
        return json.dumps({"error": f"Error during Tigris S3 upload: {type(e).__name__} - {e}"})

    print(f"Successfully uploaded images to Tigris S3. URLs: {json.dumps(s3_uploaded_urls, indent=2)}")

    # --- Stage 3: Call Synexa for Tencent Hunyuan3D API ---
    synexa_api_key = os.environ.get("SYNEXA_API_KEY")
    if not synexa_api_key:
        return json.dumps({"error": "SYNEXA_API_KEY environment variable not set. This tool requires a Synexa API Key."})

    # Hardcoded parameters for Hunyuan3D from previous step
    output_filename_hardcoded = output_filename
    seed_hardcoded = 1234
    steps_hardcoded = 20  # Corrected value
    caption_hardcoded = prompt # Use the input prompt as the caption
    shape_only_hardcoded = False
    guidance_scale_hardcoded = 5.5
    check_box_rembg_hardcoded = True
    octree_resolution_hardcoded = "256"

    output_dir_container = "/app/generated_assets" 
    os.makedirs(output_dir_container, exist_ok=True)
    
    base_output_filename = os.path.basename(output_filename_hardcoded)
    full_output_path_container = os.path.join(output_dir_container, base_output_filename)

    front_image_url_for_synexa = s3_uploaded_urls.get("front_image_url")
    back_image_url_for_synexa = s3_uploaded_urls.get("back_image_url")
    side_image_url_for_synexa = s3_uploaded_urls.get("side_image_url")

    if not front_image_url_for_synexa:
        return json.dumps({"error": "Front image URL from S3 not available for Synexa."})

    multiple_views_for_synexa = []
    if back_image_url_for_synexa:
        multiple_views_for_synexa.append(back_image_url_for_synexa)
    if side_image_url_for_synexa:
        multiple_views_for_synexa.append(side_image_url_for_synexa)
    
    headers = {
        "x-api-key": synexa_api_key,
        "Content-Type": "application/json"
    }
    payload_input = {
        "seed": seed_hardcoded,
        "image": front_image_url_for_synexa,
        "steps": steps_hardcoded,
        "caption": caption_hardcoded,
        "shape_only": shape_only_hardcoded,
        "guidance_scale": guidance_scale_hardcoded,
        "check_box_rembg": check_box_rembg_hardcoded,
        "octree_resolution": octree_resolution_hardcoded
    }
    if multiple_views_for_synexa:
        payload_input["multiple_views"] = multiple_views_for_synexa
    else:
        payload_input["multiple_views"] = [] 

    payload = {
        "model": "tencent/hunyuan3d-2",
        "input": payload_input
    }
    
    predictions_url = "https://api.synexa.ai/v1/predictions"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client: 
            print(f"Hunyuan3D: Sending POST to {predictions_url} with payload: {json.dumps(payload, indent=2)}")
            post_response = await client.post(predictions_url, headers=headers, json=payload)
            
            if post_response.status_code == 402:
                return json.dumps({"error": "Error generating 3D model: Synexa API returned '402 Payment Required'. Check your Synexa account/billing."})
            post_response.raise_for_status()
            
            post_response_data = post_response.json()
            print(f"Hunyuan3D: POST response data: {post_response_data}")

            prediction_id = post_response_data.get("id") or post_response_data.get("prediction_id")
            if not prediction_id:
                return json.dumps({"error": f"Error: Could not get prediction ID from Synexa POST response: {post_response_data}"})
            
            prediction_status_url = f"{predictions_url}/{prediction_id}"
            
            max_polling_attempts = 70 
            polling_delay_seconds = 15 
            
            for attempt in range(max_polling_attempts):
                print(f"Hunyuan3D: Polling attempt {attempt + 1}/{max_polling_attempts} for {prediction_status_url}")
                await asyncio.sleep(polling_delay_seconds) 
                get_response = await client.get(prediction_status_url, headers=headers)
                
                if get_response.status_code == 402: 
                    return json.dumps({"error": "Error generating 3D model: Synexa API returned '402 Payment Required' during status check."})
                get_response.raise_for_status()
                get_response_data = get_response.json()
                print(f"Hunyuan3D: GET response data (attempt {attempt+1}): {get_response_data}")
                
                status = get_response_data.get("status")
                if status == "succeeded":
                    output_files = get_response_data.get("output")
                    model_url_to_download = None
                    url_to_white_mesh = None
                    url_to_textured_mesh = None

                    if isinstance(output_files, list) and len(output_files) > 0:
                        for file_info in output_files:
                            current_url = None
                            if isinstance(file_info, str) and file_info.startswith("http"):
                                current_url = file_info
                            elif isinstance(file_info, dict) and 'url' in file_info and file_info['url'].startswith("http"):
                                current_url = file_info['url']
                            
                            if current_url:
                                if "_white_mesh.glb" in current_url:
                                    url_to_white_mesh = current_url
                                elif "_textured_mesh.glb" in current_url:
                                    url_to_textured_mesh = current_url
                                    model_url_to_download = current_url # Prioritize textured mesh for download
                        
                        # Fallback if specific names aren't found but we still want to try downloading something
                        if not model_url_to_download:
                            for file_info in output_files: # Check again for any valid .glb or .obj
                                fallback_url = None
                                if isinstance(file_info, str) and (file_info.endswith(".glb") or file_info.endswith(".obj")):
                                    fallback_url = file_info
                                elif isinstance(file_info, dict) and 'url' in file_info and \
                                     (file_info['url'].endswith(".glb") or file_info['url'].endswith(".obj")):
                                    fallback_url = file_info['url']
                                if fallback_url and fallback_url.startswith("http"):
                                    model_url_to_download = fallback_url
                                    if not url_to_textured_mesh: # If textured wasn't specifically identified
                                        url_to_textured_mesh = fallback_url # Assume this might be it
                                    break 

                    elif isinstance(output_files, str) and output_files.startswith("http"): # Single string output
                        model_url_to_download = output_files
                        if "_textured_mesh.glb" in output_files:
                            url_to_textured_mesh = output_files
                        elif "_white_mesh.glb" in output_files:
                             url_to_white_mesh = output_files
                        # If it's a single URL and not specifically named, we assume it's the primary model
                        if not url_to_textured_mesh and not url_to_white_mesh:
                            url_to_textured_mesh = output_files


                    if model_url_to_download:
                        print(f"Hunyuan3D: Identified model URL for download: {model_url_to_download}")
                        print(f"Hunyuan3D: White mesh URL: {url_to_white_mesh}")
                        print(f"Hunyuan3D: Textured mesh URL: {url_to_textured_mesh}")
                        
                        model_data_response = await client.get(model_url_to_download, timeout=300.0)
                        model_data_response.raise_for_status()
                        with open(full_output_path_container, "wb") as f:
                            f.write(model_data_response.content)
                        
                        host_output_path = os.path.expanduser(os.path.join("~", "generated_assets", base_output_filename))
                        
                        return json.dumps({
                            "path_to_mesh": host_output_path,
                            "url_to_white_mesh": url_to_white_mesh,
                            "url_to_textured_mesh": url_to_textured_mesh,
                            # "synexa_details": get_response_data # Optionally include for debugging
                        }, indent=4)
                    else:
                        return json.dumps({"error": f"Error: Model URL not found or invalid in Synexa 'succeeded' response. Output: {output_files}"})
                elif status in ["failed", "canceled"]:
                    error_detail = get_response_data.get("error", "No error details provided.")
                    return json.dumps({"error": f"Error: Synexa 3D model generation {status}. Details: {error_detail}"})
                elif status in ["starting", "processing"]:
                    pass 
                else: 
                    return json.dumps({"error": f"Error: Unknown Synexa prediction status '{status}'. Full response: {get_response_data}"})

            return json.dumps({"error": f"Error: Synexa 3D model generation timed out after {max_polling_attempts * polling_delay_seconds} seconds. Last status: {status}"})

    except httpx.HTTPStatusError as e:
        error_response_text = e.response.text if e.response else "No response body"
        print(f"Hunyuan3D: HTTPStatusError: {e}, Response status: {e.response.status_code}, Response text: {error_response_text}")
        if e.response.status_code == 402 or "402 Payment Required" in error_response_text:
             return json.dumps({"error": "Error generating 3D model: Synexa API returned '402 Payment Required'. Check your Synexa account/billing."})
        return json.dumps({"error": f"Error during Synexa API call: HTTP {e.response.status_code}. Response: {error_response_text}"})
    except httpx.RequestError as e: 
        print(f"Hunyuan3D: RequestError: {e}")
        return json.dumps({"error": f"Error connecting to Synexa API: {e}"})
    except Exception as e: 
        print(f"Hunyuan3D: Unexpected error: {type(e).__name__} - {e}")
        return json.dumps({"error": f"An unexpected error occurred during 3D model generation: {type(e).__name__} - {e}"})

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)